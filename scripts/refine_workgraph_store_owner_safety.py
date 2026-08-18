from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


py_path = Path("entroly/work_graph_store.py")
py = py_path.read_text(encoding="utf-8")
old = '''    @staticmethod
    def _local_lock_owner_alive(token: str) -> bool:
        parts = token.rsplit(":", 2)
        if len(parts) != 3 or parts[0] != socket.gethostname():
            return False
        try:
            pid = int(parts[1])
        except ValueError:
            return False
        if pid <= 0:
            return False
        if pid == os.getpid():
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
'''
new = '''    @staticmethod
    def _local_lock_owner_state(token: str) -> str:
        """Return alive/dead/unknown without guessing about another host.

        Malformed tokens are treated as reclaimable legacy/corrupt metadata once
        the stale-age checks pass. A syntactically valid foreign-host token is
        different: this process cannot prove the remote owner died, so recovery
        fails closed instead of risking concurrent writers on a shared volume.
        """

        parts = token.rsplit(":", 2)
        if len(parts) != 3:
            return "dead"
        host, pid_text, _nonce = parts
        if host != socket.gethostname():
            return "unknown"
        try:
            pid = int(pid_text)
        except ValueError:
            return "dead"
        if pid <= 0:
            return "dead"
        if pid == os.getpid():
            return "alive"
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "dead"
        except PermissionError:
            return "alive"
        except OSError:
            return "unknown"
        return "alive"
'''
py = replace_once(py, old, new, "python owner state")
py = replace_once(
    py,
    '''        first_token = self._lock_token()
        if self._local_lock_owner_alive(first_token):
            return False
''',
    '''        first_token = self._lock_token()
        if self._local_lock_owner_state(first_token) != "dead":
            return False
''',
    "python first owner guard",
)
py = replace_once(
    py,
    '''        second_token = self._lock_token()
        if second_token != first_token or self._local_lock_owner_alive(second_token):
            return False
''',
    '''        second_token = self._lock_token()
        if second_token != first_token or self._local_lock_owner_state(second_token) != "dead":
            return False
''',
    "python second owner guard",
)
py_path.write_text(py, encoding="utf-8")

js_path = Path("entroly-wasm/js/work_graph_store.js")
js = js_path.read_text(encoding="utf-8")
old = '''function localLockOwnerAlive(token) {
  const parts = String(token || '').split(':');
  if (parts.length < 3) return false;
  const uuid = parts.pop();
  const pidText = parts.pop();
  const host = parts.join(':');
  if (!uuid || host !== os.hostname()) return false;
  const pid = Number(pidText);
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  if (pid === process.pid) return true;
  try { process.kill(pid, 0); return true; }
  catch (error) {
    if (error && error.code === 'EPERM') return true;
    return false;
  }
}
'''
new = '''function localLockOwnerState(token) {
  const parts = String(token || '').split(':');
  if (parts.length < 3) return 'dead';
  const nonce = parts.pop();
  const pidText = parts.pop();
  const host = parts.join(':');
  if (!nonce) return 'dead';
  if (host !== os.hostname()) return 'unknown';
  const pid = Number(pidText);
  if (!Number.isSafeInteger(pid) || pid <= 0) return 'dead';
  if (pid === process.pid) return 'alive';
  try { process.kill(pid, 0); return 'alive'; }
  catch (error) {
    if (error && error.code === 'ESRCH') return 'dead';
    if (error && error.code === 'EPERM') return 'alive';
    return 'unknown';
  }
}
'''
js = replace_once(js, old, new, "node owner state")
js = replace_once(
    js,
    '''    const firstToken = this.readLockToken();
    if (localLockOwnerAlive(firstToken)) return false;
''',
    '''    const firstToken = this.readLockToken();
    if (localLockOwnerState(firstToken) !== 'dead') return false;
''',
    "node first owner guard",
)
js = replace_once(
    js,
    '''    const secondToken = this.readLockToken();
    if (secondToken !== firstToken || localLockOwnerAlive(secondToken)) return false;
''',
    '''    const secondToken = this.readLockToken();
    if (secondToken !== firstToken || localLockOwnerState(secondToken) !== 'dead') return false;
''',
    "node second owner guard",
)
js_path.write_text(js, encoding="utf-8")

print("foreign-owner stale-lock safety refinement applied")
