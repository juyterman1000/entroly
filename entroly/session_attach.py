"""Least-privilege MCP attachments for existing agent sessions.

An attachment is a local capability grant, not access to a client's private
conversation database. Entroly exposes only the selected MCP tools, binds the
server to one project root, and checks the grant before every tool invocation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


ATTACHMENT_CLIENTS = ("claude", "codex", "openclaw")
_GRANT_ID_RE = re.compile(r"^att_[0-9a-f]{16}$")
_MAX_TTL_SECONDS = 30 * 24 * 60 * 60

TOOL_SCOPES: dict[str, frozenset[str]] = {
    "observe": frozenset(
        {
            "get_stats",
            "entroly_dashboard",
            "explain_context",
            "vault_status",
            "coverage_gaps",
        }
    ),
    "context": frozenset(
        {
            "optimize_context",
            "prepare_task_dream",
            "entroly_retrieve",
            "recall_relevant",
            "prefetch_related",
            "prepare_proof_guided_context",
            "smart_read",
            "repo_file_map",
        }
    ),
    "receipts": frozenset(
        {
            "create_context_receipt",
            "create_context_receipt_from_path",
            "render_context_receipt",
            "explain_receipt_omission",
            "recover_receipt_omission",
            "verify_provenance",
        }
    ),
    "verify": frozenset(
        {
            "verify_response",
            "advance_proof_guided_context",
            "verify_and_repair",
            "eicv_verify_claim",
            "eicv_suppress_hallucinations",
            "security_scan",
            "scan_for_vulnerabilities",
            "security_report",
            "analyze_codebase_health",
            "inspect_proof_guided_context",
        }
    ),
    "remember": frozenset(
        {
            "remember_fragment",
            "checkpoint_state",
            "resume_state",
        }
    ),
    "record": frozenset(
        {
            "record_outcome",
            "record_test_result",
            "record_command_exit",
            "record_ci_result",
            "record_edit_outcome",
        }
    ),
    "vault": frozenset(
        {
            "vault_query",
            "vault_search",
            "vault_write_belief",
            "vault_write_action",
            "compile_beliefs",
            "verify_beliefs",
        }
    ),
}
DEFAULT_SCOPES = ("observe", "context", "receipts", "verify")


class AttachmentError(RuntimeError):
    """A secure attachment grant could not be created or authorized."""


@dataclass(frozen=True, slots=True)
class AttachmentGrant:
    grant_id: str
    client: str
    session_id: str | None
    project_root: str
    scopes: tuple[str, ...]
    tools: tuple[str, ...]
    created_at: float
    expires_at: float
    revoked_at: float | None
    last_used_at: float | None
    use_count: int

    @property
    def status(self) -> str:
        if self.revoked_at is not None:
            return "revoked"
        if self.expires_at <= time.time():
            return "expired"
        return "active"

    def public_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status
        return payload


@dataclass(frozen=True, slots=True)
class IssuedAttachment:
    grant: AttachmentGrant
    token_file: Path
    install_commands: tuple[tuple[str, ...], ...]

    def public_payload(self) -> dict[str, object]:
        return {
            "grant": self.grant.public_payload(),
            "token_file": str(self.token_file),
            "install_commands": [list(command) for command in self.install_commands],
        }


def parse_ttl(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([smhd]?)\s*", value.lower())
    if not match:
        raise ValueError("TTL must be an integer followed by s, m, h, or d")
    amount = int(match.group(1))
    multiplier = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]
    ttl = amount * multiplier
    if not 1 <= ttl <= _MAX_TTL_SECONDS:
        raise ValueError("TTL must be between 1 second and 30 days")
    return ttl


def resolve_tool_scopes(scopes: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selected = tuple(dict.fromkeys(scope.strip().lower() for scope in scopes if scope.strip()))
    if not selected:
        selected = DEFAULT_SCOPES
    unknown = sorted(set(selected) - TOOL_SCOPES.keys())
    if unknown:
        raise ValueError(f"unknown attachment scopes: {', '.join(unknown)}")
    tools = sorted({tool for scope in selected for tool in TOOL_SCOPES[scope]})
    return selected, tuple(tools)


class AttachmentStore:
    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.attachments_dir = self.state_dir / "attachments"
        self.db_path = self.state_dir / "attachments.sqlite3"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.attachments_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS attachment_grants (
                    grant_id TEXT PRIMARY KEY,
                    client TEXT NOT NULL,
                    session_id TEXT,
                    project_root TEXT NOT NULL,
                    scopes_json TEXT NOT NULL,
                    tools_json TEXT NOT NULL,
                    token_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    revoked_at REAL,
                    last_used_at REAL,
                    use_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS attachment_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    grant_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    occurred_at REAL NOT NULL,
                    detail TEXT,
                    FOREIGN KEY (grant_id) REFERENCES attachment_grants(grant_id)
                );
                CREATE INDEX IF NOT EXISTS idx_attachment_grants_expiry
                    ON attachment_grants(expires_at);
                CREATE INDEX IF NOT EXISTS idx_attachment_events_grant
                    ON attachment_events(grant_id, occurred_at);
                """
            )

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _token_path(self, grant_id: str) -> Path:
        if not _GRANT_ID_RE.fullmatch(grant_id):
            raise AttachmentError("invalid attachment grant id")
        return self.attachments_dir / f"{grant_id}.token"

    def create(
        self,
        *,
        client: str,
        project_root: str | Path,
        scopes: Iterable[str] = DEFAULT_SCOPES,
        ttl_seconds: int = 3600,
        session_id: str | None = None,
        now: float | None = None,
    ) -> IssuedAttachment:
        normalized_client = client.strip().lower()
        if normalized_client not in ATTACHMENT_CLIENTS:
            raise ValueError(f"unsupported attachment client: {client}")
        if not 1 <= ttl_seconds <= _MAX_TTL_SECONDS:
            raise ValueError("TTL must be between 1 second and 30 days")
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"attachment project root does not exist: {root}")
        selected_scopes, tools = resolve_tool_scopes(scopes)
        normalized_session = session_id.strip()[:200] if session_id and session_id.strip() else None

        issued_at = time.time() if now is None else float(now)
        grant_id = f"att_{secrets.token_hex(8)}"
        token = secrets.token_urlsafe(32)
        token_file = self._token_path(grant_id)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(token_file, flags, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(token)
                handle.write("\n")
        except Exception:
            token_file.unlink(missing_ok=True)
            raise

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO attachment_grants (
                        grant_id, client, session_id, project_root, scopes_json,
                        tools_json, token_hash, created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        grant_id,
                        normalized_client,
                        normalized_session,
                        str(root),
                        json.dumps(selected_scopes, separators=(",", ":")),
                        json.dumps(tools, separators=(",", ":")),
                        self._token_hash(token),
                        issued_at,
                        issued_at + ttl_seconds,
                    ),
                )
                connection.execute(
                    "INSERT INTO attachment_events (grant_id, event, occurred_at) VALUES (?, ?, ?)",
                    (grant_id, "created", issued_at),
                )
        except Exception:
            token_file.unlink(missing_ok=True)
            raise

        grant = self.get(grant_id)
        return IssuedAttachment(
            grant=grant,
            token_file=token_file,
            install_commands=attachment_install_commands(grant, self.state_dir, token_file),
        )

    @staticmethod
    def _grant_from_row(row: sqlite3.Row) -> AttachmentGrant:
        return AttachmentGrant(
            grant_id=row["grant_id"],
            client=row["client"],
            session_id=row["session_id"],
            project_root=row["project_root"],
            scopes=tuple(json.loads(row["scopes_json"])),
            tools=tuple(json.loads(row["tools_json"])),
            created_at=float(row["created_at"]),
            expires_at=float(row["expires_at"]),
            revoked_at=float(row["revoked_at"]) if row["revoked_at"] is not None else None,
            last_used_at=(
                float(row["last_used_at"]) if row["last_used_at"] is not None else None
            ),
            use_count=int(row["use_count"]),
        )

    def get(self, grant_id: str) -> AttachmentGrant:
        self._token_path(grant_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM attachment_grants WHERE grant_id = ?", (grant_id,)
            ).fetchone()
        if row is None:
            raise AttachmentError("attachment grant not found")
        return self._grant_from_row(row)

    def list(self, *, include_inactive: bool = True) -> tuple[AttachmentGrant, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM attachment_grants ORDER BY created_at DESC"
            ).fetchall()
        grants = tuple(self._grant_from_row(row) for row in rows)
        if include_inactive:
            return grants
        return tuple(grant for grant in grants if grant.status == "active")

    def authorize(
        self,
        grant_id: str,
        token: str,
        *,
        tool: str | None = None,
        now: float | None = None,
    ) -> AttachmentGrant:
        checked_at = time.time() if now is None else float(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM attachment_grants WHERE grant_id = ?", (grant_id,)
            ).fetchone()
            if row is None:
                raise AttachmentError("attachment grant not found")
            grant = self._grant_from_row(row)
            if grant.revoked_at is not None:
                connection.execute(
                    """
                    INSERT INTO attachment_events (grant_id, event, occurred_at, detail)
                    VALUES (?, 'authorization_denied', ?, 'revoked')
                    """,
                    (grant_id, checked_at),
                )
                connection.commit()
                raise AttachmentError("attachment grant has been revoked")
            if grant.expires_at <= checked_at:
                connection.execute(
                    """
                    INSERT INTO attachment_events (grant_id, event, occurred_at, detail)
                    VALUES (?, 'authorization_denied', ?, 'expired')
                    """,
                    (grant_id, checked_at),
                )
                connection.commit()
                raise AttachmentError("attachment grant has expired")
            if not hmac.compare_digest(row["token_hash"], self._token_hash(token.strip())):
                connection.execute(
                    """
                    INSERT INTO attachment_events (grant_id, event, occurred_at, detail)
                    VALUES (?, 'authorization_denied', ?, 'invalid_token')
                    """,
                    (grant_id, checked_at),
                )
                connection.commit()
                raise AttachmentError("attachment token is invalid")
            if tool is not None and tool not in grant.tools:
                connection.execute(
                    """
                    INSERT INTO attachment_events (grant_id, event, occurred_at, detail)
                    VALUES (?, 'authorization_denied', ?, 'tool_outside_scope')
                    """,
                    (grant_id, checked_at),
                )
                connection.commit()
                raise AttachmentError(f"tool {tool!r} is outside the attachment scope")
            connection.execute(
                """
                UPDATE attachment_grants
                SET last_used_at = ?, use_count = use_count + 1
                WHERE grant_id = ?
                """,
                (checked_at, grant_id),
            )
        return self.get(grant_id)

    def revoke(self, grant_id: str, *, now: float | None = None) -> AttachmentGrant:
        revoked_at = time.time() if now is None else float(now)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT revoked_at FROM attachment_grants WHERE grant_id = ?", (grant_id,)
            ).fetchone()
            if row is None:
                raise AttachmentError("attachment grant not found")
            if row["revoked_at"] is None:
                connection.execute(
                    "UPDATE attachment_grants SET revoked_at = ? WHERE grant_id = ?",
                    (revoked_at, grant_id),
                )
                connection.execute(
                    """
                    INSERT INTO attachment_events (grant_id, event, occurred_at)
                    VALUES (?, 'revoked', ?)
                    """,
                    (grant_id, revoked_at),
                )
        self._token_path(grant_id).unlink(missing_ok=True)
        return self.get(grant_id)


def attachment_install_commands(
    grant: AttachmentGrant,
    state_dir: str | Path,
    token_file: str | Path,
) -> tuple[tuple[str, ...], ...]:
    name = f"entroly-{grant.grant_id}"
    server = (
        "entroly",
        "attach",
        "serve",
        "--grant-id",
        grant.grant_id,
        "--token-file",
        str(Path(token_file).resolve()),
        "--state-dir",
        str(Path(state_dir).resolve()),
    )
    source_env = f"ENTROLY_SOURCE={grant.project_root}"
    if grant.client == "claude":
        return (("claude", "mcp", "add", "--scope", "local", "-e", source_env, name, "--", *server),)
    if grant.client == "codex":
        return (("codex", "mcp", "add", name, "--env", source_env, "--", *server),)
    if grant.client == "openclaw":
        add = ["openclaw", "mcp", "add", name, "--command", "entroly"]
        for argument in server[1:]:
            add.extend(("--arg", argument))
        add.extend(("--env", source_env))
        return (tuple(add), ("openclaw", "mcp", "reload"))
    raise ValueError(f"unsupported attachment client: {grant.client}")


def attachment_remove_commands(grant: AttachmentGrant) -> tuple[tuple[str, ...], ...]:
    name = f"entroly-{grant.grant_id}"
    if grant.client == "claude":
        return (("claude", "mcp", "remove", "--scope", "local", name),)
    if grant.client == "codex":
        return (("codex", "mcp", "remove", name),)
    if grant.client == "openclaw":
        return (
            ("openclaw", "mcp", "unset", name),
            ("openclaw", "mcp", "reload"),
        )
    raise ValueError(f"unsupported attachment client: {grant.client}")


def install_attachment(
    issued: IssuedAttachment,
    *,
    store: AttachmentStore | None = None,
    runner=subprocess.run,
) -> tuple[subprocess.CompletedProcess[str], ...]:
    results: list[subprocess.CompletedProcess[str]] = []
    for command in issued.install_commands:
        try:
            result = runner(command, check=False, text=True, capture_output=True)
        except OSError as exc:
            result = subprocess.CompletedProcess(command, 127, "", str(exc))
        results.append(result)
        if result.returncode != 0:
            rollback_errors: list[str] = []
            for rollback in attachment_remove_commands(issued.grant):
                try:
                    removal = runner(rollback, check=False, text=True, capture_output=True)
                    if removal.returncode != 0:
                        rollback_errors.append(
                            f"{rollback[0]} exit {removal.returncode}: {removal.stderr.strip()}"
                        )
                except OSError as exc:
                    rollback_errors.append(str(exc))
            if store is not None:
                try:
                    store.revoke(issued.grant.grant_id)
                except AttachmentError as exc:
                    rollback_errors.append(f"grant revocation failed: {exc}")
            recovery = (
                f"rollback incomplete ({'; '.join(rollback_errors)})"
                if rollback_errors
                else "client configuration rolled back and grant revoked"
            )
            raise AttachmentError(
                f"client configuration failed ({command[0]} exit {result.returncode}): "
                f"{result.stderr.strip()}; {recovery}"
            )
    return tuple(results)


def uninstall_attachment(
    grant: AttachmentGrant,
    *,
    runner=subprocess.run,
) -> tuple[subprocess.CompletedProcess[str], ...]:
    """Remove a client entry after access has already been revoked."""
    results: list[subprocess.CompletedProcess[str]] = []
    for command in attachment_remove_commands(grant):
        try:
            result = runner(command, check=False, text=True, capture_output=True)
        except OSError as exc:
            raise AttachmentError(
                f"access is revoked, but client configuration removal failed: {exc}; "
                f"run manually: {format_command(command)}"
            ) from exc
        results.append(result)
        if result.returncode != 0:
            raise AttachmentError(
                "access is revoked, but client configuration removal failed "
                f"({command[0]} exit {result.returncode}): {result.stderr.strip()}; "
                f"run manually: {format_command(command)}"
            )
    return tuple(results)


def serve_attachment(
    *,
    grant_id: str,
    token_file: str | Path,
    state_dir: str | Path,
) -> None:
    store = AttachmentStore(state_dir)
    token_path = Path(token_file).expanduser().resolve()
    expected_path = store._token_path(grant_id).resolve()
    if token_path != expected_path:
        raise AttachmentError("attachment token file does not match the grant")
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise AttachmentError(f"attachment token file is unavailable: {exc}") from exc
    grant = store.authorize(grant_id, token)
    project_root = Path(grant.project_root)
    if not project_root.is_dir():
        raise AttachmentError("attachment project root is no longer available")

    os.environ["ENTROLY_DIR"] = str(store.state_dir)
    os.environ["ENTROLY_SOURCE"] = str(project_root)
    os.chdir(project_root)

    from entroly.server import _start_background_services, create_mcp_server

    def authorize_tool(tool: str) -> None:
        store.authorize(grant_id, token, tool=tool)

    mcp, engine = create_mcp_server(
        allowed_tools=set(grant.tools),
        authorize_tool=authorize_tool,
    )
    _start_background_services(engine)
    mcp.run()


def format_command(command: Sequence[str]) -> str:
    """Render a copyable command without invoking a platform shell."""
    return subprocess.list2cmdline(list(command))
