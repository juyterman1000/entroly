# Persistent Entroly installation

`entroly install` manages one user-scoped Entroly daemon. It does not request administrator privileges or write a system-wide service.

```bash
entroly install apply --dry-run
entroly install apply
entroly install status --json
entroly install restart
entroly install stop
entroly install start
entroly install remove
```

The platform backends are:

- Linux: `systemd --user` unit in `~/.config/systemd/user/entroly.service`.
- macOS: launchd user agent in `~/Library/LaunchAgents/io.entroly.daemon.plist`.
- Windows: least-privilege logon task named `Entroly` in Task Scheduler.

The generated command runs `python -m entroly.cli daemon`, sets `PYTHONSAFEPATH=1`, binds to loopback by default, restarts after failure, and prevents duplicate Windows task instances. `--no-proxy`, `--no-mcp`, ports, host, and quality can be declared at installation time.

## Reversibility and removal safety

Entroly records the service-definition path and SHA-256 digest in `~/.entroly/install/manifest.json`. `entroly install remove` refuses to delete a definition that changed after Entroly wrote it. Removal disables the user service and removes only the digest-matched definition and Entroly manifest; logs and other user state remain available.

`--dry-run` prints the definition and lifecycle commands without writing or executing anything. The [platform readiness matrix](platform-readiness.json) distinguishes hosted-runner verification from service-definition contract tests; it does not claim that CI registered a real desktop service.
