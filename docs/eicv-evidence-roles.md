# EICV evidence roles and code-answer limits

EICV's Evidence Support Graph (ESG) scores supplied reference text against a
claim. A task question is not a factual premise: asking "Does Vault encrypt
backups?" does not establish "Vault encrypts backups."

ESG now excludes questions recognized by WITNESS's existing English question
recognizer from its evidence atoms. It preserves ordinary declarative clauses
starting with words such as "when", "where", and "what". Recognized unpunctuated
question lines are excluded without consuming an adjacent source block, and the
short-text fallback cannot reintroduce a discarded question. It does not join
text across omitted question lines to manufacture a new premise.

Keep task instructions and questions separate from reference material whenever
possible. This filter is not a general discourse parser, multilingual question
detector, prompt-injection defense, or source-authority check. Recognition and
sentence splitting remain heuristic; a question can evade recognition.

## What a decision does not establish

`supported` means support according to the configured lexical/structural
heuristics. It is not proof of code execution semantics, source completeness,
causal use of a cited passage, or truth. `hallucinated` is a low-support detector
decision, not proof that a claim is false: correct code answers may still be
rejected. An exact source hash proves byte identity, not those semantics.

Default public SDK calls do not fit calibrators. Their e-product, e-mean and
conservative-p defaults of 1.0 are not affirmative statistical evidence. Fitted
calibration also cannot turn lexical support into a semantic code proof.

Question eligibility changes scores for affected inputs. Existing committed
benchmark results describe their recorded implementation, not an evaluation of
this change. Re-run an appropriate frozen workload and refit any relevant
calibration before making updated accuracy or risk claims.

## Research boundary

[ContextCite (NeurIPS 2024)](https://arxiv.org/html/2409.00729v2) distinguishes
contributive attribution from corroboration: a model can use a source and still
misinterpret it. [Agentic Code Reasoning](https://arxiv.org/html/2603.01896v1)
studies structured premises and code traces but reports remaining errors from
incomplete traces and assumed library behavior. Neither result makes a lexical
question filter a semantic verifier or a new research contribution.
