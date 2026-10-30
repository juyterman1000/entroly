"""Provider-bound TLS trusts the OS store, and extra roots only ever add trust.

Behind a TLS-inspecting corporate gateway (Zscaler, Netskope, ...) the company
root lives in the operating-system store. The proxy verified against certifi
alone, so every upstream call failed CERTIFICATE_VERIFY_FAILED, and
``NODE_EXTRA_CA_CERTS`` replaced the whole store with the one file it names.
"""

from __future__ import annotations

import builtins
import datetime
import socket
import ssl
import threading
from pathlib import Path

import certifi
import pytest

import entroly.proxy as proxy_module
from entroly import tls_trust
from entroly.proxy_transport_final import (
    _safe_http_client_kwargs as final_client_kwargs,
)
from entroly.proxy_transport_safe import (
    _safe_http_client_kwargs as safe_client_kwargs,
)

BUNDLE_VARS = ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS", "ENTROLY_TLS_TRUST")


@pytest.fixture(autouse=True)
def _clean_trust_env(monkeypatch):
    for name in BUNDLE_VARS:
        monkeypatch.delenv(name, raising=False)


def _without_truststore(monkeypatch) -> None:
    """Force the OpenSSL-system fallback, whose loaded roots are inspectable."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "truststore":
            raise ImportError("truststore hidden for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _certifi_count() -> int:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile=certifi.where())
    return len(context.get_ca_certs())


# ── policy ───────────────────────────────────────────────────────────────


def test_default_policy_uses_the_os_store() -> None:
    context, policy = tls_trust.upstream_ssl_context()
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname is True
    pytest.importorskip("truststore")
    assert policy == "os-store"


def test_certifi_roots_stay_trusted_alongside_the_system_store(monkeypatch) -> None:
    _without_truststore(monkeypatch)
    context, policy = tls_trust.upstream_ssl_context()
    assert policy == "openssl-system"
    # Nothing that verified against certifi before may stop verifying.
    assert len(context.get_ca_certs()) >= _certifi_count()


def test_node_extra_ca_certs_adds_roots_instead_of_replacing_them(
    monkeypatch, tmp_path: Path
) -> None:
    ca = _make_ca(tmp_path)
    _without_truststore(monkeypatch)
    baseline = len(tls_trust.upstream_ssl_context()[0].get_ca_certs())

    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", str(ca.cert_path))
    context, policy = tls_trust.upstream_ssl_context()

    assert policy.endswith("+node_extra_ca_certs")
    assert len(context.get_ca_certs()) == baseline + 1


def test_explicit_bundle_is_the_whole_trust_set(monkeypatch, tmp_path: Path) -> None:
    ca = _make_ca(tmp_path)
    monkeypatch.setenv("SSL_CERT_FILE", str(ca.cert_path))
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", certifi.where())

    context, policy = tls_trust.upstream_ssl_context()

    assert policy == "exclusive-bundle"
    assert len(context.get_ca_certs()) == 1


def test_bundled_mode_restores_the_previous_certifi_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ENTROLY_TLS_TRUST", "bundled")
    context, policy = tls_trust.upstream_ssl_context()
    assert policy == "bundled"
    assert len(context.get_ca_certs()) == _certifi_count()

    # The rollback switch keeps NODE_EXTRA_CA_CERTS additive: restoring the
    # bug that made it the whole store would break every public host.
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", str(_make_ca(tmp_path).cert_path))
    context, policy = tls_trust.upstream_ssl_context()
    assert policy == "bundled+node_extra_ca_certs"
    assert len(context.get_ca_certs()) == _certifi_count() + 1


def test_explicit_bundle_may_be_a_hashed_directory(monkeypatch, tmp_path: Path) -> None:
    # httpx accepted a CA directory for verify=; the policy must too.
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path))
    context, policy = tls_trust.upstream_ssl_context()
    assert policy == "exclusive-bundle"
    assert isinstance(context, ssl.SSLContext)


def test_unreadable_extra_bundle_does_not_break_the_proxy(monkeypatch, caplog) -> None:
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", "/nonexistent/corporate-root.pem")
    context, policy = tls_trust.upstream_ssl_context()
    assert isinstance(context, ssl.SSLContext)
    assert "node_extra" not in policy
    assert "Ignoring unreadable CA bundle from NODE_EXTRA_CA_CERTS" in caplog.text


# ── every provider client builder uses the policy ────────────────────────


@pytest.mark.parametrize(
    "build",
    [proxy_module._http_client_kwargs, safe_client_kwargs, final_client_kwargs],
    ids=["proxy", "transport_safe", "transport_final"],
)
def test_provider_clients_verify_with_the_shared_policy(build) -> None:
    kwargs = build()
    assert isinstance(kwargs["verify"], ssl.SSLContext)


@pytest.mark.parametrize(
    "build", [safe_client_kwargs, final_client_kwargs], ids=["transport_safe", "transport_final"]
)
def test_trust_policy_does_not_enable_ambient_proxy_routing(build, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", str(_make_ca(tmp_path).cert_path))
    assert build()["trust_env"] is False


# ── a real handshake ─────────────────────────────────────────────────────


def test_handshake_with_a_gateway_root(monkeypatch, tmp_path: Path) -> None:
    """A gateway-signed host fails without its root and verifies with it."""
    ca = _make_ca(tmp_path)
    host, port, stop = _serve_tls(ca, tmp_path)
    try:
        monkeypatch.setenv("ENTROLY_TLS_TRUST", "bundled")
        bundled, _ = tls_trust.upstream_ssl_context()
        with pytest.raises(ssl.SSLCertVerificationError):
            _handshake(bundled, host, port)

        monkeypatch.delenv("ENTROLY_TLS_TRUST")
        default_context, _ = tls_trust.upstream_ssl_context()
        with pytest.raises(ssl.SSLCertVerificationError):
            _handshake(default_context, host, port)  # verification is still on

        monkeypatch.setenv("NODE_EXTRA_CA_CERTS", str(ca.cert_path))
        extra_context, _ = tls_trust.upstream_ssl_context()
        _handshake(extra_context, host, port)
    finally:
        stop()


# ── helpers ──────────────────────────────────────────────────────────────


class _CA:
    def __init__(self, key, cert, cert_path: Path) -> None:
        self.key = key
        self.cert = cert
        self.cert_path = cert_path


def _make_ca(tmp_path: Path) -> _CA:
    pytest.importorskip("cryptography")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Entroly Test Gateway Root")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    path = tmp_path / f"gateway-root-{id(cert)}.pem"
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return _CA(key, cert, path)


def _serve_tls(ca: _CA, tmp_path: Path):
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.datetime.now(datetime.timezone.utc)
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
        .issuer_name(ca.cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca.key.public_key()),
            critical=False,
        )
        .sign(ca.key, hashes.SHA256())
    )
    cert_file = tmp_path / "leaf.pem"
    key_file = tmp_path / "leaf.key"
    cert_file.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_file, key_file)

    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(0.2)
    port = listener.getsockname()[1]
    stopped = threading.Event()

    def serve() -> None:
        while not stopped.is_set():
            try:
                conn, _ = listener.accept()
            except (socket.timeout, OSError):
                continue
            try:
                conn.settimeout(5)
                with server_context.wrap_socket(conn, server_side=True):
                    pass
            except (ssl.SSLError, OSError):
                pass  # the client rejected us; that is the point of some cases

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    def stop() -> None:
        stopped.set()
        thread.join(timeout=5)
        listener.close()

    return "localhost", port, stop


def _handshake(context: ssl.SSLContext, host: str, port: int) -> None:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        with context.wrap_socket(sock, server_hostname=host):
            pass
