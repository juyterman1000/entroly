# Entroly Context Check action

This composite action creates a replayable Context Commit, compares its selected
files with files changed in a pull request, writes a bounded Markdown summary,
and uploads the schema-defined JSON, Markdown, and Context Commit artifacts.

```yaml
jobs:
  context-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: juyterman1000/entroly/.github/actions/entroly-context-check@main
        with:
          task: ${{ github.event.pull_request.title }}
          token-budget: 8000
          fail-on-risk: high
```

For non-pull-request events, pass a fetched `base-ref`. Without a comparison
base, the action still creates a Context Commit but marks changed-file coverage
as `not_measured` and risk as `unknown`. A configured threshold treats unknown
risk conservatively.

Changed-file recall is retrospective coverage evidence. It does not prove task
correctness, identify every file that should have been read, or prove that a
model used the selected context.
