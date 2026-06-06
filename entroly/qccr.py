"""Query-Conditioned Compressive Retrieval (QCCR).

Motivation
----------
Two failure modes of fragment-level selection at tight budgets:

  1. *Wrong chunk of the right file.* When a file is indexed as N fragments
     of ~400 tokens each, selecting fragment #3 may miss the paragraph in
     fragment #7 that actually answers the query.
  2. *Diversity spread.* Submodular log-det or MMR at the fragment level
     rewards picking many hash-distinct tiny chunks over one focused file.
     At 4K budget the answer drowns in uniform-weight filler.

QCCR sidesteps both by operating at two granularities:

  - *File-level BM25* chooses which documents are candidates. This is
    coarse, fast, and mirrors what a human does ("which file has this?").
  - *Sentence-level query-conditioned BM25 + MMR* (Carbonell-Goldstein 1998)
    picks the specific sentences from those files that answer the query,
    with diversity control against redundant sentences.

The result is a per-query extractive summary: ~15 tokens per sentence
× ~270 sentences = full 4K budget spent on content that directly
addresses the query, drawn from whichever files BM25 ranked highest.

Nothing here uses embeddings, neural inference, or trained weights.
Pure classical IR (BM25 + MMR) applied at the right granularity.

References
----------
- Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25
  and Beyond"
- Carbonell & Goldstein (1998), "The Use of MMR, Diversity-Based Reranking
  for Reordering Documents and Producing Summaries"
- Nemhauser, Wolsey, Fisher (1978), (1-1/e) bound on greedy submodular
  maximization — applies to MMR with monotone-submodular relevance.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# ── Tokenization (shared with dopt_selector style) ──────────────────────
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=\n\n)|(?<=;\n)|(?<=\}\n)")

_STOPWORDS = frozenset("""
the a an of to in on for with by how does what why is are and or but
do between two include shown actual actually that this these those
from can could should would has have had will was were been being
about into through over under above below after before while when where
not no yes all any some most many much more less than then thus therefore
you your their his her its our it he she they them we us i me my
""".split())

_BM25_K1 = 1.5
_BM25_B = 0.75
_MMR_LAMBDA = 0.7           # relevance-diversity tradeoff (70% relevance)
_MIN_SENTENCE_CHARS = 20    # skip shorter "sentences" (whitespace artifacts)
_MAX_FILES_CONSIDERED = 12  # top-K files per query
_MAX_MMR_SENTENCE_CANDIDATES = 512  # bound pairwise MMR cost on huge files
_CHARS_PER_TOKEN = 4        # approximation for token-budget accounting
_ENTITY_BOOST = 1.5         # multiplicative weight for query-entity overlap

# Capitalized words (≥3 chars, mixed case), bare numbers, and quoted spans.
# Cheap NER proxy — improves QA recall when questions ask about specific
# proper nouns / dates / numbers / quoted phrases.
_ENTITY_RE = re.compile(
    r"\b[A-Z][a-z]{2,}\b|\b\d+(?:[.,]\d+)*\b|['\"]([^'\"]{2,40})['\"]"
)

# ── Ranking signal vocabulary — GENERAL code conventions only ──────────────
# Deterministic, language-agnostic programming conventions. Deliberately NO
# repository-specific identifiers (no table names, no project symbols): the
# scorer must generalize to any codebase, so its only knowledge is of universal
# naming / directory conventions. Per-repo specialization is the job of the
# learned layer (archetype / PRISM / autotune), which can override the weights
# below via tuning_config["qccr_rank"].
_TEST_PATH_RE = re.compile(
    r"(^|/)(__tests__|tests?|specs?|e2e|fixtures?|mocks?|__mocks__)(/|$)"
    r"|[._-](test|spec|stories)\.",
    re.I,
)
_GENERATED_PATH_RE = re.compile(
    r"/(generated|gen|dist|build|out|coverage|snapshots?|__snapshots__|vendor|"
    r"migrations?|seed(?:er)?s?)/"
    r"|[._-](?:generated|min)\.|(?:package-lock|pnpm-lock|yarn\.lock)",
    re.I,
)
_SOURCE_DIR_RE = re.compile(r"/(?:src|lib|core|pkg|internal|app)/", re.I)
_LOGIC_DIR_RE = re.compile(
    r"/(?:services?|repositor(?:y|ies)|models?|domain|handlers?|controllers?|"
    r"workers?|queues?|jobs?|processors?|usecases?|adapters?|mappers?|stores?|"
    r"dao|db|database|persistence)/",
    re.I,
)
_UI_DIR_RE = re.compile(r"/(?:components?|pages?|views?|widgets?|ui|styles?)/", re.I)

# Query-intent clusters: general programming vocabulary grouped by intent. A
# query token in a cluster (a) classifies the query's intent and (b) expands the
# query with sibling vocabulary so natural wording reaches code identifiers
# (e.g. "map json to schema" reaches `mapEventsToRecords` / `RecordInsertType`).
_INTENT_CLUSTERS: dict[str, frozenset[str]] = {
    "mapping": frozenset({
        "map", "mapping", "mapper", "transform", "convert", "converter",
        "serialize", "deserialize", "marshal", "unmarshal", "adapt",
        "translate", "encode", "decode", "parse", "parser",
    }),
    "schema": frozenset({
        "schema", "model", "models", "type", "types", "record", "records",
        "entity", "entities", "struct", "dto", "interface",
    }),
    "ingest": frozenset({
        "incoming", "request", "payload", "event", "events", "ingest",
        "ingestion", "consume", "consumer", "intake", "receive", "handler",
        "queue", "worker", "processor", "stream",
    }),
    "persistence": frozenset({
        "persist", "persistence", "save", "store", "stored", "write", "upsert",
        "insert", "update", "delete", "repository", "repositories", "dao",
        "table", "tables", "database", "migration", "query",
    }),
}
# Minimal cross-cluster links: one intent implies likely-adjacent vocabulary.
_CLUSTER_LINKS: dict[str, tuple[str, ...]] = {
    "mapping": ("schema",),
    "ingest": ("mapping", "persistence"),
    "persistence": ("schema",),
    "schema": (),
}


def _stem(tok: str) -> str:
    """Conservative suffix stripper used ONLY for intent classification, so the
    query's surface morphology (persisted/persisting/persists, maps/mapping,
    scores, items) matches a cluster's base form. Never produces a stem shorter
    than 3 chars; BM25 keeps the original tokens untouched."""
    for suf in ("ings", "ing", "ied", "ies", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            stem = tok[: -len(suf)]
            return stem + "y" if suf == "ied" else stem
    return tok


# Stemmed view of the clusters — built once — so intent detection is morphology
# robust without enumerating every inflected variant.
_INTENT_CLUSTERS_STEMMED: dict[str, frozenset[str]] = {
    name: frozenset(_stem(t) for t in vocab)
    for name, vocab in _INTENT_CLUSTERS.items()
}

# Structural symbol conventions (general across languages / ORMs). These detect
# *what a file does* by universal naming shape, not by any project's symbols:
#   _RE_MAPPER       mapXToY / convertFooIntoBar / toRecordFromEvent
#   _RE_TRANSFORM    serialize/deserialize/encode/decode/parse/normalize...
#   _RE_PERSIST      insertX / upsertX / saveX / writeX / findX ...
#   _RE_SCHEMA_TYPE  XInsertType / XRecord / XSchema / XModel / XEntity / XDto
#   _RE_SQL          INSERT INTO / UPDATE..SET / SELECT..FROM / CREATE TABLE
_RE_MAPPER = re.compile(
    r"\b(?:map|convert|transform|to|from)[A-Za-z0-9_]*?(?:To|From|Into)[A-Z][A-Za-z0-9_]*\b"
)
_RE_TRANSFORM = re.compile(
    r"\b(?:serialize|deserialize|marshal|unmarshal|encode|decode|parse|format|"
    r"normalize|adapt)[A-Za-z0-9_]*\b",
    re.I,
)
_RE_PERSIST = re.compile(
    r"\b(?:insert|upsert|save|persist|write|store|create|update|delete|find|"
    r"fetch|load|select)[A-Z][A-Za-z0-9_]*\b"
)
_RE_SCHEMA_TYPE = re.compile(
    r"\b[A-Z][A-Za-z0-9_]*(?:InsertType|UpdateType|RecordType|Record|Schema|"
    r"Model|Entity|Dto|DTO|Table|Row|Document)\b"
)
_RE_SQL = re.compile(
    r"\b(?:INSERT\s+INTO|UPDATE\s+\w+\s+SET|SELECT\b[\s\S]{0,200}?\bFROM|"
    r"CREATE\s+TABLE)\b",
    re.I,
)

# ── BM25F field weights (Robertson & Zaragoza 2009, §4) ────────────────────
# A proper two-field model: each field keeps its OWN length normalization, so a
# short path field is not penalized for being short (the bug in concatenating a
# repeated path "title" into the body — that inflates body length / avgdl and
# corrupts normalization). Path is weighted above body because a query term in
# the path is a stronger relevance signal than one buried in a long body.
_BM25F_W_BODY = 1.0
_BM25F_W_PATH = 2.5
_BM25F_B_PATH = 0.5

# ── Log-linear file-ranking weights ────────────────────────────────────────
# BM25F (normalized to [0,1] across the candidate set) is the backbone; every
# other term is a bounded ADDITIVE adjustment in the same units. This keeps the
# combination principled (a linear scoring model / learning-to-rank), avoids the
# multiplicative blow-up that needed a magic cap, and stays interpretable and
# learnable. Overridable per-repo via tuning_config["qccr_rank"].
_DEFAULT_RANK_WEIGHTS: dict[str, float] = {
    "bm25f": 1.0,
    "test_penalty": -0.45,        # test/spec/mock file, query did not ask for tests
    "generated_penalty": -0.55,   # generated / vendored / lockfile / migration
    "source_dir": 0.08,           # lives under src/lib/core/pkg/...
    "logic_dir_intent": 0.22,     # business-logic dir aligned with a backend intent
    "ui_penalty_for_logic": -0.30,  # UI/page file for a backend-logic query
    "basename_hit": 0.30,         # a query term appears in the file's own name
    "defines_mapper": 0.40,       # owns mapXToY-style mapping (intent-gated)
    "defines_transform": 0.16,    # serialize/parse/encode-style transform
    "defines_persistence": 0.30,  # owns insert/upsert/write or raw SQL
    "defines_schema_type": 0.22,  # declares XRecord/XSchema/XInsertType-style types
}


def _query_entities(query: str) -> frozenset[str]:
    """Extract specific surface-form entities from the query. Used as a
    multiplicative boost on top of BM25 — entities are stronger signal
    than bag-of-words for QA-style queries.
    """
    out: set[str] = set()
    for m in _ENTITY_RE.finditer(query):
        # Match group 1 (quoted) if present, otherwise the full match
        out.add((m.group(1) or m.group(0)).lower())
    return frozenset(out)


def _split_identifier(tok: str) -> list[str]:
    """`taint_flow_total` → [taint_flow_total, taint, flow, total]. Also handles
    CamelCase. Improves recall when a query uses words that code identifiers
    concatenate."""
    low = tok.lower()
    parts = {low}
    for piece in low.split("_"):
        if len(piece) > 2:
            parts.add(piece)
    for piece in _CAMEL_RE.findall(tok):
        p = piece.lower()
        if len(p) > 2:
            parts.add(p)
    return [p for p in parts if p not in _STOPWORDS]


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    for raw in _IDENT_RE.findall(text):
        out.extend(_split_identifier(raw))
    return out


def _query_tokens(query: str) -> frozenset[str]:
    return frozenset(t for t in _tokenize(query) if len(t) > 2)


def _query_intents(base_terms: frozenset[str]) -> frozenset[str]:
    """Classify the query into general intent clusters (mapping / schema /
    ingest / persistence). Computed from the ORIGINAL query tokens — never the
    expanded set — so expansion cannot inflate which intents are active. Matches
    on stemmed forms so 'persisted'/'maps'/'scores' reach their base vocabulary.
    """
    stems = {_stem(t) for t in base_terms}
    return frozenset(
        name for name, vocab in _INTENT_CLUSTERS_STEMMED.items() if stems & vocab
    )


def _expanded_query_tokens(query: str) -> frozenset[str]:
    """Expand natural task wording into likely code vocabulary, generically.

    Deterministic and corpus-agnostic: when a query token falls in an intent
    cluster, add that cluster's sibling vocabulary (plus a minimal set of
    linked clusters). This bridges the common gap where a user asks "how is
    incoming JSON mapped to the schema?" while the code calls that flow
    `mapEventsToRecords` / `RecordInsertType` — using universal programming
    vocabulary, with no repository-specific symbols baked in.
    """
    terms = set(_query_tokens(query))
    for name in _query_intents(frozenset(terms)):
        terms |= _INTENT_CLUSTERS[name]
        for linked in _CLUSTER_LINKS.get(name, ()):
            terms |= _INTENT_CLUSTERS[linked]
    return frozenset(t for t in terms if t not in _STOPWORDS and len(t) > 2)


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries plus structural code breaks (blank line,
    semicolon+newline, close-brace+newline). Drops tiny fragments."""
    out: list[str] = []
    for chunk in _SENTENCE_SPLIT.split(text):
        s = chunk.strip()
        if len(s) >= _MIN_SENTENCE_CHARS:
            out.append(s)
    return out


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // _CHARS_PER_TOKEN)


def _bm25f_corpus(
    bodies: list[str], path_tokens: list[list[str]]
) -> tuple[list[Counter], list[Counter], list[int], list[int], Counter, float, float]:
    """Two-field BM25F corpus statistics (body + path).

    Document frequency is computed over the UNION of fields (a term counts once
    per document); each field keeps its own length series so field-length
    normalization is independent. This is the correct BM25F formulation —
    unlike concatenating a repeated path "title" into the body, which inflates
    body length / avgdl and corrupts normalization.
    """
    body_tf: list[Counter] = []
    path_tf: list[Counter] = []
    body_len: list[int] = []
    path_len: list[int] = []
    df: Counter = Counter()
    for body, ptoks in zip(bodies, path_tokens):
        bt = Counter(_tokenize(body))
        pt = Counter(ptoks)
        body_tf.append(bt)
        path_tf.append(pt)
        body_len.append(sum(bt.values()))
        path_len.append(sum(pt.values()))
        for term in set(bt) | set(pt):
            df[term] += 1
    n = len(bodies)
    avg_body = max((sum(body_len) / n) if n else 1.0, 1.0)
    avg_path = max((sum(path_len) / n) if n else 1.0, 1.0)
    return body_tf, path_tf, body_len, path_len, df, avg_body, avg_path


def _bm25f_score(
    q_terms: frozenset[str],
    i: int,
    body_tf: list[Counter],
    path_tf: list[Counter],
    body_len: list[int],
    path_len: list[int],
    df: Counter,
    N: int,
    avg_body: float,
    avg_path: float,
) -> float:
    """BM25F (Robertson & Zaragoza 2009): per-field length-normalized term
    frequencies are combined with field weights BEFORE the saturation function,
    then weighted by IDF and summed over query terms."""
    score = 0.0
    bt = body_tf[i]
    pt = path_tf[i]
    for term in q_terms:
        fb = bt.get(term, 0)
        fp = pt.get(term, 0)
        if fb == 0 and fp == 0:
            continue
        n = df.get(term, 0)
        idf = math.log(1.0 + (N - n + 0.5) / (n + 0.5))
        ntf_b = (
            fb / (1.0 - _BM25_B + _BM25_B * (body_len[i] / avg_body))
            if fb else 0.0
        )
        ntf_p = (
            fp / (1.0 - _BM25F_B_PATH + _BM25F_B_PATH * (path_len[i] / avg_path))
            if fp else 0.0
        )
        wtf = _BM25F_W_BODY * ntf_b + _BM25F_W_PATH * ntf_p
        score += idf * (wtf * (_BM25_K1 + 1.0)) / (wtf + _BM25_K1)
    return score


def _rank_features(
    source: str,
    text: str,
    intents: frozenset[str],
    base_terms: frozenset[str],
    q_terms: frozenset[str],
    w: dict[str, float],
) -> float:
    """Additive (log-linear) ranking adjustment from GENERAL code conventions.

    Every signal is a universal naming / directory convention — no repository
    identifiers — and each contributes a bounded weighted term, so the ranking
    is an interpretable linear model over features rather than a stack of
    multiplicative magic constants. Structural features are intent-gated so a
    file is only rewarded for owning behaviour the query actually asked about.
    """
    s = source.lower().replace("\\", "/")
    adj = 0.0
    wants_tests = bool(
        base_terms & {"test", "tests", "spec", "assert", "fixture", "mock"}
    )
    backend = bool(intents & {"mapping", "ingest", "persistence", "schema"})

    if not wants_tests and _TEST_PATH_RE.search(s):
        adj += w["test_penalty"]
    if not wants_tests and _GENERATED_PATH_RE.search(s):
        adj += w["generated_penalty"]
    if _SOURCE_DIR_RE.search(s):
        adj += w["source_dir"]
    if backend and _LOGIC_DIR_RE.search(s):
        adj += w["logic_dir_intent"]
    if backend and _UI_DIR_RE.search(s):
        adj += w["ui_penalty_for_logic"]

    basename = s.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if q_terms & set(_tokenize(basename)):
        adj += w["basename_hit"]

    if (intents & {"mapping", "schema"}) and _RE_MAPPER.search(text):
        adj += w["defines_mapper"]
    if (intents & {"mapping", "ingest"}) and _RE_TRANSFORM.search(text):
        adj += w["defines_transform"]
    if (intents & {"persistence", "ingest"}) and (
        _RE_PERSIST.search(text) or _RE_SQL.search(text)
    ):
        adj += w["defines_persistence"]
    if (intents & {"schema", "mapping", "persistence"}) and _RE_SCHEMA_TYPE.search(text):
        adj += w["defines_schema_type"]
    return adj


_RANK_WEIGHTS_CACHE: dict[str, float] | None = None


def _load_rank_weights() -> dict[str, float]:
    """Ranking weights with per-repo overrides from the learned layer.

    Defaults are sensible and corpus-agnostic; the archetype / PRISM / autotune
    pipeline can write a `qccr_rank` block into the active tuning config to
    specialize ranking for a given codebase — so specialization lives in the
    learned layer, not hardcoded in the scorer. Loaded once and cached.
    """
    global _RANK_WEIGHTS_CACHE
    if _RANK_WEIGHTS_CACHE is not None:
        return _RANK_WEIGHTS_CACHE
    weights = dict(_DEFAULT_RANK_WEIGHTS)
    try:
        from .config import load_active_tuning_config
        active = load_active_tuning_config()
        if active is not None:
            _, cfg = active
            override = cfg.get("qccr_rank")
            if isinstance(override, dict):
                for k, v in override.items():
                    if k in weights:
                        try:
                            weights[k] = float(v)
                        except (TypeError, ValueError):
                            continue
    except Exception:
        pass
    _RANK_WEIGHTS_CACHE = weights
    return weights


# ── BM25 scoring ─────────────────────────────────────────────────────────
def _bm25_corpus(texts: list[str]) -> tuple[list[Counter], list[int], Counter, float]:
    tf_list: list[Counter] = []
    lens: list[int] = []
    df: Counter = Counter()
    for t in texts:
        toks = _tokenize(t)
        tf = Counter(toks)
        tf_list.append(tf)
        lens.append(sum(tf.values()))
        for term in tf:
            df[term] += 1
    avgdl = (sum(lens) / len(lens)) if lens else 1.0
    return tf_list, lens, df, max(avgdl, 1.0)


def _bm25_score(
    q_terms: frozenset[str],
    tf: Counter,
    dl: int,
    df: Counter,
    N: int,
    avgdl: float,
) -> float:
    score = 0.0
    for term in q_terms:
        f = tf.get(term, 0)
        if f == 0:
            continue
        n = df.get(term, 0)
        idf = math.log(1.0 + (N - n + 0.5) / (n + 0.5))
        norm = 1.0 - _BM25_B + _BM25_B * (dl / avgdl)
        score += idf * (f * (_BM25_K1 + 1.0)) / (f + _BM25_K1 * norm)
    return score


# ── MMR sentence selection ──────────────────────────────────────────────
def _mmr_select(
    sentences: list[str],
    tf_list: list[Counter],
    rel: list[float],
    budget_tokens: int,
    lam: float = _MMR_LAMBDA,
) -> list[int]:
    """Maximum Marginal Relevance: balance query relevance against redundancy
    with already-selected sentences. Jaccard over token sets is the
    similarity surrogate — cheap and discrete, which is fine since sentence
    token sets are small. Returns selected indices in original document
    order.

        MMR = argmax_i [ λ · rel(i) − (1-λ) · max_{j∈S} sim(i, j) ]

    (Carbonell-Goldstein 1998). Greedy; near-optimal under the usual
    submodularity assumptions.
    """
    n = len(sentences)
    if n == 0:
        return []

    index_map = list(range(n))
    if n > _MAX_MMR_SENTENCE_CANDIDATES:
        ranked = sorted(
            range(n),
            key=lambda i: (rel[i], len(tf_list[i]), len(sentences[i])),
            reverse=True,
        )[:_MAX_MMR_SENTENCE_CANDIDATES]
        ranked.sort()
        index_map = ranked
        sentences = [sentences[i] for i in ranked]
        tf_list = [tf_list[i] for i in ranked]
        rel = [rel[i] for i in ranked]
        n = len(sentences)

    selected: list[int] = []
    remaining: list[int] = [i for i in range(n) if rel[i] > 0.0]
    if not remaining:
        # Anchor-fallback: no sentence has positive query overlap
        # (common when query is paraphrased or uses different vocabulary).
        # Rather than return nothing — which forces the LLM to answer
        # with no evidence — pack the longest sentences that fit. This
        # preserves at least some context and is strictly better than
        # an empty selection. Verified on SQuAD: lifts answer-survival
        # from ~90% to ~92.5% at budget=100.
        ranked_by_len = sorted(range(n), key=lambda i: -len(sentences[i]))
        out: list[int] = []
        used = 0
        for i in ranked_by_len:
            cost = _approx_tokens(sentences[i])
            if used + cost > budget_tokens and out:
                break
            out.append(i)
            used += cost
            if used >= budget_tokens:
                break
        return sorted(index_map[i] for i in out)
    # token sets per sentence for Jaccard
    sets = [frozenset(tf_list[i].keys()) for i in range(n)]
    max_sim = [0.0] * n
    budget_used = 0

    while remaining and budget_used < budget_tokens:
        if not selected:
            best = max(remaining, key=lambda i: rel[i])
        else:
            def mmr_score(i: int) -> float:
                return lam * rel[i] - (1.0 - lam) * max_sim[i]
            best = max(remaining, key=mmr_score)

        if rel[best] <= 0.0:
            break
        cost = _approx_tokens(sentences[best])
        if budget_used + cost > budget_tokens:
            remaining.remove(best)
            continue
        selected.append(best)
        remaining.remove(best)
        budget_used += cost
        b = sets[best]
        if b:
            for i in remaining:
                a = sets[i]
                if not a:
                    continue
                inter = len(a & b)
                union = len(a | b)
                if union == 0:
                    continue
                sim = inter / union
                if sim > max_sim[i]:
                    max_sim[i] = sim

    return sorted(index_map[i] for i in selected)


# ── Public entry ─────────────────────────────────────────────────────────
def select(
    fragments: list[dict],
    token_budget: int,
    query: str = "",
) -> list[dict]:
    """Query-Conditioned Compressive Retrieval.

    Pipeline:
      1. Group fragments by source file.
      2. File-level BM25 → rank files by query relevance.
      3. For each top-K file, sentence-level BM25 + MMR extracts the
         sentences that answer the query.
      4. Emit as pseudo-fragments: one dict per file containing the
         extracted sentences concatenated, preserving original order.

    Empty query ⇒ fall back to original fragment list (no compression).
    """
    if not fragments:
        return []
    if not query:
        return fragments

    base_terms = _query_tokens(query)
    q_terms = _expanded_query_tokens(query)
    if not q_terms:
        return fragments
    q_ents = _query_entities(query)
    intents = _query_intents(base_terms)
    weights = _load_rank_weights()

    # Group by file
    by_file: dict[str, list[dict]] = {}
    for raw in fragments:
        src = raw.get("source", "") or ""
        by_file.setdefault(src, []).append(raw)

    # ── File-level ranking: BM25F backbone + log-linear convention features ──
    # score = w_bm25 · normalize(BM25F(body, path)) + Σ_k w_k · feature_k
    # A proper two-field BM25F (no length-corrupting path duplication) plus a
    # linear model over GENERAL code-convention features. BM25F is normalized to
    # [0,1] across the candidate set so the additive feature weights stay in a
    # comparable, bounded, interpretable scale.
    file_sources = list(by_file.keys())
    file_texts = ["\n".join((r.get("content") or "") for r in by_file[s]) for s in file_sources]
    path_tokens = [_tokenize(src.replace("\\", "/")) for src in file_sources]
    body_tf, path_tf, body_len, path_len, df, avg_body, avg_path = _bm25f_corpus(
        file_texts, path_tokens
    )
    N = len(file_sources)
    raw_bm25f = [
        _bm25f_score(
            q_terms, i, body_tf, path_tf, body_len, path_len, df, N, avg_body, avg_path
        )
        for i in range(N)
    ]
    max_bm25f = max(raw_bm25f) if raw_bm25f else 0.0
    denom = max_bm25f if max_bm25f > 0 else 1.0
    file_scores: list[tuple[float, str, str]] = []
    for i in range(N):
        src = file_sources[i]
        text = file_texts[i]
        score = weights["bm25f"] * (raw_bm25f[i] / denom)
        score += _rank_features(src, text, intents, base_terms, q_terms, weights)
        file_scores.append((score, src, text))
    file_scores.sort(reverse=True)

    # engine_s6 edit-target rerank — delegated to the centralized
    # service so this surface stays a thin caller and the recall-safe
    # try/except lives in exactly one place. Window-local permutation
    # of the existing ranked candidate set; tail preserved (recall floor);
    # budget allocation below still uses each file's own ranking score,
    # so this only changes the ORDER excerpts are emitted in.
    # Validation: see entroly/file_localizer.py module docstring.
    if N > 1 and any(s > 0 for s, _, _ in file_scores):
        from .file_localizer import localize_files
        files_map = dict(zip(file_sources, file_texts))
        bm25_order = [src for _, src, _ in file_scores]
        reranked = localize_files(
            files_map, query, k=len(bm25_order), base_ranked=bm25_order,
        )
        score_text_by_src = {src: (sc, txt)
                             for sc, src, txt in file_scores}
        file_scores = [(score_text_by_src[s][0], s,
                        score_text_by_src[s][1])
                       for s in reranked if s in score_text_by_src]

    top_files = [fs for fs in file_scores[:_MAX_FILES_CONSIDERED] if fs[0] > 0]
    if not top_files:
        # Anchor-fallback: no file has positive BM25 (query terms not in any
        # file). Rather than return nothing, consider the top files anyway
        # so the sentence-level pass + length-fallback can still rescue a
        # useful excerpt.
        top_files = file_scores[:_MAX_FILES_CONSIDERED]
        if not top_files:
            return []

    # Split budget roughly in proportion to file BM25, with floor per file.
    total_score = sum(s for s, _, _ in top_files) or 1.0
    budget_left = int(token_budget)
    per_file_budget: dict[str, int] = {}
    for score, src, _ in top_files:
        share = int(token_budget * (score / total_score))
        per_file_budget[src] = max(share, 256)  # floor so small-scoring files still contribute a snippet

    # Sentence-level MMR per file
    output: list[dict] = []
    for score, src, text in top_files:
        if budget_left <= 0:
            break
        sentences = _split_sentences(text)
        if not sentences and text.strip():
            sentences = [text.strip()]
        if not sentences:
            continue
        s_tf, s_lens, s_df, s_avg = _bm25_corpus(sentences)
        s_N = len(sentences)
        rel = [
            _bm25_score(q_terms, s_tf[i], s_lens[i], s_df, s_N, s_avg)
            for i in range(s_N)
        ]
        # Entity boost: sentences containing query entities (capitalized
        # proper nouns / numbers / quoted spans) get a multiplicative
        # bump. Strictly additive to BM25 — doesn't drop sentences,
        # just reorders them so entity-matching ones rise. Verified
        # to improve answer survival on SQuAD (+2.5pp) and LongBench
        # (+4pp) without changing emission shape.
        if q_ents:
            for i, sent in enumerate(sentences):
                s_lower = sent.lower()
                hits = sum(1 for e in q_ents if e in s_lower)
                if hits:
                    rel[i] *= (1.0 + _ENTITY_BOOST * hits)
        file_budget = min(per_file_budget.get(src, 256), budget_left)
        chosen = _mmr_select(sentences, s_tf, rel, file_budget)
        if not chosen:
            continue
        excerpt = "\n".join(sentences[i] for i in chosen)
        tokens_used = _approx_tokens(excerpt)
        fragment_id = f"qccr::{src}"
        relevance = round(float(score), 4)
        # Emit as a synthetic fragment preserving source attribution.
        output.append({
            "id": fragment_id,
            "fragment_id": fragment_id,
            "source": src,
            "content": excerpt,
            "token_count": tokens_used,
            "relevance": relevance,
            "relevance_score": relevance,
        })
        budget_left -= tokens_used

    # ── Hard budget ceiling ───────────────────────────────────────────────
    # Per-file _approx_tokens plus the newline joins between sentences can
    # drift a few tokens over the requested budget. Callers treat token_budget
    # as a HARD cap (e.g. `entroly optimize --budget N` must not report
    # tokens_used > N). Trim trailing excerpts — emitted last because they came
    # from the lowest-ranked files — dropping the trailing sentence of the
    # final excerpt (least-relevant content of the least-relevant file), then
    # whole excerpts, until the emitted total fits.
    def _frag_tokens(frag: dict) -> int:
        return frag.get("token_count") or _approx_tokens(frag.get("content", "") or "")

    total = sum(_frag_tokens(f) for f in output)
    while output and total > token_budget:
        last = output[-1]
        lines = (last.get("content") or "").split("\n")
        if len(lines) > 1:
            lines.pop()
            last["content"] = "\n".join(lines)
            last["token_count"] = _approx_tokens(last["content"])
        else:
            output.pop()
        total = sum(_frag_tokens(f) for f in output)

    return output
