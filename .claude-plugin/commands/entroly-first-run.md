---
description: Show what Entroly selected from this repository, what it omitted, and how to recover the omissions.
---

Show the user what Entroly does to their own repository, using their real code.

1. Run `entroly compile` against the project's primary source directory, then
   `entroly simulate` scoped to that directory.
2. Report three things, in this order:
   - **Selected** — the fragments that were kept, with their files.
   - **Omitted** — the fragments that were left out, with their files. Name
     them. Do not summarise them as a count.
   - **Recovery** — for each omitted fragment, the content-addressed handle
     that retrieves the exact original bytes. Demonstrate one live recovery.
3. Close by inviting the user to pick any omitted fragment and recover it.

Do not report a savings percentage, a compression ratio, or a token-reduction
figure. The percentage is determined by the configured budget before selection
runs, so it measures the budget rather than the selection. The claim worth
making is that nothing was lost irrecoverably, and that claim is checkable in
front of the user — which is the entire point.

If the native engine is missing, say so plainly and state that selection has
not been query-conditioned, rather than reporting a figure that looks earned.
