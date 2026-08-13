# AI Traffic Value

Entroly exposes an executive value view at:

```text
http://127.0.0.1:9377/traffic-value
```

The machine-readable surface is:

```text
GET /traffic-value.json
```

The view keeps the same durable value counters Entroly already records and shows them as rolling windows:

- Today
- 7 days
- 30 days
- 60 days
- 90 days
- All time

`All time` is the persistent cumulative total and survives proxy restarts. The rolling views do not reset at calendar week/month boundaries, so a buyer can compare a stable recent window with the lifetime value Entroly has accumulated.

## Evidence contract

The main dollar number is **estimated provider input value avoided**. It is calculated from provider-bound token reduction observed by Entroly and the configured model input rate. It is intentionally not labelled as provider invoice savings or a measured counterfactual.

Local SDK, MCP, npm, and other reductions remain token-only because Entroly cannot prove that those outputs were sent to a paid provider.

Pricing provenance is displayed with every snapshot, including the active source and `as_of` date.

The dashboard reads the existing persistent `ValueTracker`; it does not create a second savings ledger or double-count Traffic Receipt events.
