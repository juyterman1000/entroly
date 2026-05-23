# entroly

Compatibility package for npm users.

This package installs the current Node/WebAssembly Entroly runtime by depending
on [`entroly-wasm`](https://www.npmjs.com/package/entroly-wasm), then exposes a
short `entroly` binary that delegates to `entroly-wasm`.

## Install

```bash
npm install -g entroly
```

Equivalent direct package:

```bash
npm install -g entroly-wasm
```

## Usage

```bash
entroly serve
entroly optimize 8000 "fix the auth bug"
entroly health
entroly demo
```

For Python users, use the PyPI package instead:

```bash
pip install entroly
entroly go
```
