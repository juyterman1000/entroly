"""One TLS trust policy for every provider-bound connection.

Corporate networks (Zscaler, Netskope, Palo Alto, Cisco Secure Access, ...)
inspect TLS by re-signing every certificate with a company root that IT
installs in the operating-system trust store -- macOS Keychain, the Windows
certificate store, or the distribution's CA directory on Linux. Browsers,
``curl`` and Claude Code read that store, so they keep working.

The proxy did not. httpx verifies against certifi's bundled Mozilla roots, so
behind such a gateway every upstream call failed with
``CERTIFICATE_VERIFY_FAILED`` unless the user happened to export a CA bundle
variable, and ``NODE_EXTRA_CA_CERTS`` -- which by name and convention adds
roots -- replaced the whole trust store with the one file it names.

Policy, in order:

1. ``REQUESTS_CA_BUNDLE`` or ``SSL_CERT_FILE`` set: trust exactly that file
   (or hashed directory). An operator who names a bundle is choosing the trust
   set; that has always been the behaviour and stays so.
2. Otherwise trust the operating-system store (via ``truststore`` when it is
   importable, else OpenSSL's system paths) **and** certifi's bundle, so no
   host that verifies today stops verifying.
3. ``NODE_EXTRA_CA_CERTS`` is added on top of (2), never instead of it.

``ENTROLY_TLS_TRUST=bundled`` drops the OS store again and trusts certifi's
bundle (plus ``NODE_EXTRA_CA_CERTS``) -- the previous behaviour, less the bug
that made the extra-roots file the entire store. Building a context reads
local files only; nothing here touches the network.
"""

from __future__ import annotations

import logging
import os
import ssl
from typing import Any

logger = logging.getLogger(__name__)

EXCLUSIVE_BUNDLE_VARS = ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")
ADDITIVE_BUNDLE_VARS = ("NODE_EXTRA_CA_CERTS",)
TRUST_MODE_VAR = "ENTROLY_TLS_TRUST"


def exclusive_ca_bundle() -> str | None:
    """The operator-named bundle that should be the whole trust set, if any."""
    for name in EXCLUSIVE_BUNDLE_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _trust_mode() -> str:
    mode = os.environ.get(TRUST_MODE_VAR, "system").strip().lower()
    return mode if mode in {"system", "bundled"} else "system"


def _certifi_path() -> str | None:
    try:
        import certifi
    except ImportError:  # pragma: no cover - certifi ships with httpx
        return None
    return certifi.where()


def _os_store_context() -> tuple[ssl.SSLContext, str]:
    try:
        import truststore
    except ImportError:
        return ssl.create_default_context(), "openssl-system"
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return context, "os-store"


def _add_locations(context: ssl.SSLContext, path: str, *, source: str) -> bool:
    try:
        context.load_verify_locations(cafile=path)
    except (OSError, ssl.SSLError) as exc:
        # An unreadable extra bundle must not take the proxy down; the base
        # trust set still verifies everything it verified before.
        logger.warning("Ignoring unreadable CA bundle from %s: %s", source, exc)
        return False
    return True


def _exclusive_context(location: str) -> ssl.SSLContext:
    # httpx accepted a directory here as well as a file; keep accepting both.
    if os.path.isdir(location):
        return ssl.create_default_context(capath=location)
    return ssl.create_default_context(cafile=location)


def upstream_ssl_context() -> tuple[ssl.SSLContext, str]:
    """Return ``(context, policy)`` for provider-bound HTTPS.

    ``policy`` names what was chosen, for diagnostics.
    """
    exclusive = exclusive_ca_bundle()
    if exclusive:
        return _exclusive_context(exclusive), "exclusive-bundle"
    certifi_path = _certifi_path()
    if _trust_mode() == "bundled":
        # Exactly what httpx builds for its default ``verify=True``.
        context = ssl.create_default_context(cafile=certifi_path)
        policy = "bundled"
    else:
        context, policy = _os_store_context()
        if certifi_path:
            _add_locations(context, certifi_path, source="certifi")
    for name in ADDITIVE_BUNDLE_VARS:
        value = os.environ.get(name, "").strip()
        if value and _add_locations(context, value, source=name):
            policy += f"+{name.lower()}"
    return context, policy


def apply_upstream_verify(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Set ``verify`` on httpx client kwargs according to the policy."""
    kwargs["verify"], _policy = upstream_ssl_context()
    return kwargs
