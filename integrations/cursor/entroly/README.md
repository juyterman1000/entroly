# Entroly for Cursor

Cursor's native `beforeSubmitPrompt` response can allow or block a prompt but
does not have an output field for task context. Cursor also supports Claude
Code's `UserPromptSubmit` hook and nested `hookSpecificOutput` response. Entroly
uses that compatibility path.

Install the hook in a project:

```console
entroly activation install --host cursor --project .
```

Then enable **Include third-party Plugins, Skills, and other configs** under
Cursor Settings > Rules, Skills, Subagents. The account must have the feature.
Until `entroly activation status --json` shows an effective receipt, treat the
integration as unobserved.

The installer merges only Entroly's `UserPromptSubmit` entry into
`.claude/settings.local.json` and backs up an existing file. Disable the entry
while retaining a recoverable backup:

```console
entroly activation uninstall --host cursor --project .
```

An activation receipt proves local context selection. It does not prove that
provider traffic traversed Entroly or that tokens or cost were saved.
