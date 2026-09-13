# Entroly for Kiro

This bundle runs Entroly before every submitted prompt in Kiro IDE 1.x and
Kiro CLI 3.x. Install Entroly, then run this command in the project root:

```console
entroly activation install --host kiro --project .
```

Kiro automatically loads `.kiro/hooks/*.json` when the session starts. The
installer refuses to overwrite an existing file unless `--force` is supplied;
forced replacement creates a timestamped backup. Disable the managed hook
without deleting it:

```console
entroly activation uninstall --host kiro --project .
```

The command exits successfully even when local selection fails, so an Entroly
failure does not block the user's task. Successful stdout is added to Kiro's
agent context. Check execution evidence with:

```console
entroly activation status --json
```

An activation receipt proves that the hook ran and records selected local
context. It does not prove provider token or cost savings without a matched
baseline.
