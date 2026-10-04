# MCP desktop bundle

`manifest.json` is the reviewed source for the desktop MCP bundle. Its launch
configuration invokes the installed Entroly package through `uvx`; the archive
contains this manifest rather than a vendored Python environment.

Build from the repository root:

```bash
python scripts/build_mcpb.py
```

The output is `dist/entroly.mcpb`. ZIP metadata is deterministic, and version
changes rebuild the bundle through the same helper. Build outputs belong in
release artifacts and remain ignored by Git.
