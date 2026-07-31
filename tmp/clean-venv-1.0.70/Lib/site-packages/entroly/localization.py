"""Tier-0 code localization — zero-dependency, parameter-free.

Problem: given an issue (symptom language) find the repo file(s) to
edit (code language). Pure lexical BM25 is structurally weak here —
vocabulary mismatch — measured ≈0.10–0.20 hit@10 on real SWE-bench.

v2 fusion = a deterministic **evidence-precedence tier** (files with
UNAMBIGUOUS structural certainty — explicit S1 extraction, or a symbol
defined in exactly one file — ranked first) THEN **RRF(k=60)**
(Reciprocal Rank Fusion, Cormack et al. SIGIR 2009) over the fuzzy
signals. RRF is parameter-light (k=60 standard), and empirically
outperforms ISR on this signal regime. Content-BM25 is always a signal,
so the recall floor is >= plain BM25; the evidence tier + S1/S2/S4 are
pure upside. No model weights, no runtime, no network.

Signals
-------
S1  Explicit-location extraction — tracebacks `File "...py"`, dotted
    modules `a.b.c`→`a/b/c.py`|`/__init__.py`, backticked / Camel /
    snake symbols. Near-ground-truth when present; BM25 ignores it.
S2  Structure-graph PRF — symbol⇒defining-file map weighted by inverse
    file-frequency (rare symbol localizes; common = noise); seed =
    files defining issue symbols; 1-hop import-graph expansion.
S3  Dual-field BM25 — content field + symbol/path field.
S4  RM3 pseudo-relevance feedback — the zero-dep semantic bridge for
    vocabulary mismatch: expand the query with terms drawn from the
    repo's own top-retrieved code, then re-retrieve. M/E/λ are the
    standard textbook defaults (10 / 20 / 0.5), NOT tuned on any
    benchmark — so RM3 adds semantic recall without an embedding model
    and without an artifact surface.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)", re.M)
_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.M)
_IMPORT_FROM_RE = re.compile(r"^\s*from\s+([\w.]+)\s+import\b", re.M)
_IMPORT_RE = re.compile(r"^\s*import\s+([\w.]+(?:\s*,\s*[\w.]+)*)", re.M)
_TRACEBACK_FILE_RE = re.compile(r'File "([^"]+?\.py)"')
_PATHLIKE_RE = re.compile(r"\b([\w/]+\.py)\b")
_DOTTED_RE = re.compile(r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*){1,6})\b")
_BACKTICK_RE = re.compile(r"`([^`]{2,80})`")
_IDENT_RE = re.compile(r"\b([A-Za-z_]\w{3,})\b")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")

# Compact English + Python/code stopwords — RM3 must not expand the
# query with structural noise ("self", "return", "the", ...).
_STOP = frozenset("""
a an the of to in on at by for and or not is are was were be been being
this that these those it its as with from into out up down if else elif
def class return import self none true false try except raise finally
with yield lambda pass break continue global nonlocal assert del while
for in print str int float list dict set tuple bool object type len
i e g eg ie etc do does did has have had can could should would will
""".split())


def _tok(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _split_ident(name: str) -> list[str]:
    """snake_case / CamelCase → component tokens (lowercased)."""
    parts = re.split(r"[_\W]+", name)
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        out.extend(re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z0-9]+|\d+", p)
                   or [p])
    return [o.lower() for o in out if o]


class _BM25:
    """Okapi BM25 over pre-tokenised docs (k1=1.5, b=0.75)."""

    __slots__ = ("ids", "docs", "tf", "idf", "avgdl", "k1", "b", "_pos")

    def __init__(self, corpus: list[tuple[str, list[str]]]):
        self.k1, self.b = 1.5, 0.75
        self.ids = [c[0] for c in corpus]
        self.docs = [c[1] for c in corpus]
        self.tf = [Counter(d) for d in self.docs]
        self._pos = {fid: i for i, fid in enumerate(self.ids)}
        n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / n) if n else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {
            w: math.log((n - f + 0.5) / (f + 0.5) + 1.0)
            for w, f in df.items()
        }

    def _term_score(self, w: str, i: int) -> float:
        c = self.tf[i]
        f = c.get(w)
        if not f:
            return 0.0
        dl = len(self.docs[i])
        return self.idf[w] * f * (self.k1 + 1) / (
            f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9)))

    def scores(self, qweights: dict[str, float]) -> list[tuple[float, str]]:
        q = [(t, wt) for t, wt in qweights.items() if t in self.idf]
        out: list[tuple[float, str]] = []
        for i in range(len(self.docs)):
            s = 0.0
            for t, wt in q:
                if t in self.tf[i]:
                    s += wt * self._term_score(t, i)
            out.append((s, self.ids[i]))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    def ranking(self, query_terms: list[str]) -> list[str]:
        if not self.docs:
            return []
        qw: dict[str, float] = defaultdict(float)
        for t in query_terms:
            qw[t] += 1.0
        return [fid for _, fid in self.scores(qw)]

    def ranking_weighted(self, qweights: dict[str, float]) -> list[str]:
        if not self.docs or not qweights:
            return []
        return [fid for _, fid in self.scores(qweights)]


def _module_to_paths(mod: str) -> list[str]:
    base = mod.replace(".", "/")
    return [base + ".py", base + "/__init__.py"]


class Tier0Localizer:
    """Parameter-free, zero-dependency file localizer.

    Build once over `{path: content}`; call `rank(issue, k)`.
    """

    # Literature-standard RM3 constants — NOT tuned on any benchmark.
    RM3_M = 10        # feedback documents
    RM3_E = 20        # expansion terms
    RM3_LAMBDA = 0.5  # original vs. expansion mixing

    def __init__(self, files: dict[str, str], *, build_rank_index: bool = True):
        self.files = files
        self.paths = list(files)
        self._path_set = set(self.paths)
        self._content_corpus: list[tuple[str, list[str]]] = []
        self._symbol_corpus: list[tuple[str, list[str]]] = []
        self.sym_def: dict[str, set[str]] = defaultdict(set)
        self.imports: dict[str, set[str]] = defaultdict(set)
        self.imported_by: dict[str, set[str]] = defaultdict(set)

        for path, content in files.items():
            defs = set(_DEF_RE.findall(content)) | set(
                _CLASS_RE.findall(content))
            for s in defs:
                self.sym_def[s].add(path)
            sym_field = _split_ident(path.replace("/", "_").rsplit(".", 1)[0])
            for s in defs:
                sym_field += _split_ident(s)
            if build_rank_index:
                self._symbol_corpus.append((path, sym_field))
                self._content_corpus.append((path, _tok(content)))

        if build_rank_index:
            for path, content in files.items():
                mods: list[str] = list(_IMPORT_FROM_RE.findall(content))
                for grp in _IMPORT_RE.findall(content):
                    mods += [m.strip() for m in grp.split(",")]
                for mod in mods:
                    for cand in _module_to_paths(mod.strip().lstrip(".")):
                        if cand in self._path_set:
                            self.imports[path].add(cand)
                            self.imported_by[cand].add(path)
                            break

        self._bm25_content = _BM25(self._content_corpus)
        self._bm25_symbol = _BM25(self._symbol_corpus)
        self._sym_iff = {
            s: 1.0 / len(fs) for s, fs in self.sym_def.items() if fs
        }

    @classmethod
    def for_edit_rerank(cls, files: dict[str, str]) -> "Tier0Localizer":
        """Build only the path/symbol state needed by ``rerank_edit_target``."""
        return cls(files, build_rank_index=False)

    # ── S1: explicit location extraction ──────────────────────────────

    def _s1(self, issue: str) -> list[str]:
        ranked: list[str] = []
        seen: set[str] = set()

        def add(p: str):
            if p in self._path_set and p not in seen:
                seen.add(p)
                ranked.append(p)

        for raw in _TRACEBACK_FILE_RE.findall(issue) + _PATHLIKE_RE.findall(
                issue):
            raw = raw.lstrip("./")
            if raw in self._path_set:
                add(raw)
            else:
                for p in self.paths:
                    if p.endswith("/" + raw) or p.endswith(raw):
                        add(p)
                        break
        for dotted in _DOTTED_RE.findall(issue):
            for cand in _module_to_paths(dotted):
                if cand in self._path_set:
                    add(cand)
        return ranked

    # ── S2: structure-graph pseudo-relevance feedback ─────────────────

    def _issue_symbols(self, issue: str) -> list[str]:
        syms: list[str] = []
        for m in _BACKTICK_RE.findall(issue):
            syms += _IDENT_RE.findall(m)
        syms += _IDENT_RE.findall(issue)
        return [s for s in syms if s in self.sym_def]

    def _s2(self, issue: str) -> list[str]:
        syms = self._issue_symbols(issue)
        if not syms:
            return []
        score: dict[str, float] = defaultdict(float)
        seeds: set[str] = set()
        for s in syms:
            w = self._sym_iff.get(s, 0.0)
            for f in self.sym_def.get(s, ()):
                score[f] += w
                seeds.add(f)
        for f in list(seeds):
            for nb in self.imports.get(f, ()):
                score[nb] += 0.25 * score.get(f, 0.0)
            for nb in self.imported_by.get(f, ()):
                score[nb] += 0.25 * score.get(f, 0.0)
        return [fid for fid, _ in sorted(
            score.items(), key=lambda x: (-x[1], x[0])) if score]

    # ── S4: RM3 pseudo-relevance feedback (zero-dep semantic bridge) ──

    def _rm3_query(self, qtok: list[str]) -> dict[str, float]:
        """Standard RM3: relevance model from the top-M BM25 docs,
        interpolated with the original query. Parameter-free (textbook
        M/E/λ). Returns a weighted query for `ranking_weighted`."""
        bm = self._bm25_content
        q0: dict[str, float] = defaultdict(float)
        for t in qtok:
            q0[t] += 1.0
        ranked = bm.scores(q0)
        feedback = [(fid, sc) for sc, fid in ranked[: self.RM3_M] if sc > 0.0]
        if not feedback:
            return q0
        z = sum(sc for _, sc in feedback) or 1.0
        rel: dict[str, float] = defaultdict(float)
        for fid, sc in feedback:
            i = bm._pos[fid]
            doc = bm.docs[i]
            if not doc:
                continue
            dl = len(doc)
            pdq = sc / z                       # P(d|q) ∝ retrieval score
            for term, c in bm.tf[i].items():
                if (len(term) < 3 or term in _STOP or term.isdigit()
                        or term not in bm.idf):
                    continue
                rel[term] += pdq * (c / dl)    # Σ P(d|q)·P(t|d)
        if not rel:
            return q0
        top = sorted(rel.items(), key=lambda x: (-x[1], x[0]))[: self.RM3_E]
        rz = sum(w for _, w in top) or 1.0
        lam = self.RM3_LAMBDA
        out: dict[str, float] = defaultdict(float)
        qz = sum(q0.values()) or 1.0
        for t, w in q0.items():               # original query mass (λ)
            out[t] += lam * (w / qz)
        for t, w in top:                      # expansion mass (1−λ)
            out[t] += (1.0 - lam) * (w / rz)
        return out

    # ── S5: Coherent-Evidence Match (exact contiguous identity) ───────
    # The decisive precision signal. Dense embeddings BLUR exact token
    # order; SWE-bench issues quote code blocks / error strings /
    # tracebacks verbatim. Scoring a file by the IDF-weighted longest
    # CONTIGUOUS token run of that evidence in the file is laser-precise
    # exactly where embeddings are weak. (This is the WITNESS extractive
    # longest-contiguous-run primitive, proven this session, reused for
    # retrieval.) Parameter-free: k=4 shingle, IDF data-derived.

    _SHINGLE = 4

    def _evidence_spans(self, issue: str) -> list[list[str]]:
        spans: list[list[str]] = []
        for blk in re.findall(r"```(?:[\w.+-]*)\n?(.*?)```", issue, re.S):
            spans.append(_tok(blk))
        for blk in re.findall(r"~~~+\n?(.*?)~~~+", issue, re.S):
            spans.append(_tok(blk))
        for q in re.findall(r'"([^"\n]{8,200})"', issue):
            spans.append(_tok(q))
        for q in re.findall(r"'([^'\n]{8,200})'", issue):
            spans.append(_tok(q))
        for m in _BACKTICK_RE.findall(issue):
            spans.append(_tok(m))
        for fr in re.findall(r'File "[^"]+", line \d+,? in \w+\s*\n\s*(.+)',
                             issue):
            spans.append(_tok(fr))
        return [s for s in spans if len(s) >= 2]

    def _idf(self, t: str) -> float:
        return self._bm25_content.idf.get(t, 0.0)

    @staticmethod
    def _is_contiguous_sub(needle: list[str], hay: list[str]) -> bool:
        n = len(needle)
        if not n or n > len(hay):
            return False
        first = needle[0]
        for i in range(len(hay) - n + 1):
            if hay[i] == first and hay[i:i + n] == needle:
                return True
        return False

    def _s5(self, issue: str) -> tuple[list[str], list[str]]:
        """Returns (soft_ranking, exact_containment_hits).

        exact = a code/error span (>=`_SHINGLE` tokens) that appears as
        a CONTIGUOUS token substring of exactly that file → deterministic
        certainty (no tuned threshold). soft = IDF-weighted shared-shingle
        score, anchored on each span's rarest token so only the few files
        containing that rare token are scanned (precision + speed)."""
        bm = self._bm25_content
        spans = self._evidence_spans(issue)
        if not spans:
            return [], []
        score: dict[str, float] = defaultdict(float)
        exact: list[str] = []
        exact_seen: set[str] = set()
        for span in spans:
            span = span[:160]
            cand = [(self._idf(t), t) for t in set(span) if t in bm.idf]
            if not cand:
                continue
            anchor = max(cand)[1]                       # rarest token
            cand_idx = [i for i in range(len(bm.docs))
                        if anchor in bm.tf[i]]
            if not cand_idx:
                continue
            shingles = {tuple(span[j:j + self._SHINGLE])
                        for j in range(max(0, len(span) - self._SHINGLE + 1))}
            sh_w = {sh: max(self._idf(t) for t in sh) for sh in shingles}
            big = len(span) >= self._SHINGLE
            for i in cand_idx:
                doc = bm.docs[i]
                dl = len(doc)
                if dl == 0:
                    continue
                dshing = {tuple(doc[j:j + self._SHINGLE])
                          for j in range(max(0, dl - self._SHINGLE + 1))}
                hit = sum(sh_w[sh] for sh in shingles if sh in dshing)
                if hit > 0.0:
                    fid = bm.ids[i]
                    score[fid] += hit
                    if (big and fid not in exact_seen
                            and self._is_contiguous_sub(span, doc)):
                        exact_seen.add(fid)
                        exact.append(fid)
        soft = [fid for fid, _ in sorted(
            score.items(), key=lambda x: (-x[1], x[0]))]
        return soft, exact

    # ── Fusion (ISR — Inverse Square Rank, Mourão et al. 2014) ────────
    # Σ 1/rankᵢ² — parameter-FREE (no k), top-heavy so a precise signal
    # at rank-1 dominates (rank1=1.0, r2=0.25, r3=0.11), the regime our
    # S1/S2/S5 signals are in. NOT tuned on any benchmark.

    @staticmethod
    def _isr(rankings: list[list[str]]) -> dict[str, float]:
        agg: dict[str, float] = defaultdict(float)
        for r in rankings:
            for rank, fid in enumerate(r):
                agg[fid] += 1.0 / ((rank + 1) * (rank + 1))
        return agg

    def _innermost_traceback_file(self, issue: str) -> str | None:
        """Deepest `File "X", line N` frame (the fix site far more often
        than outer frames) resolved to EXACTLY ONE existing file."""
        frames = _TRACEBACK_FILE_RE.findall(issue)
        for raw in reversed(frames):                # innermost = last
            raw = raw.lstrip("./")
            if raw in self._path_set:
                return raw
            matches = [p for p in self.paths
                       if p.endswith("/" + raw) or p.endswith(raw)]
            if len(matches) == 1:                   # unambiguous only
                return matches[0]
        return None

    def rank(self, issue: str, k: int = 20) -> list[str]:
        """v4 — PURE fusion, NO override tier.

        The override/certainty tier was empirically the cause of the
        v2→v3 regressions (prepending a fallible signal + ISR
        amplification destroys recall when it errs). Removed entirely.
        Every signal now only VOTES (rank fusion is robust precisely
        because no signal can override). ISR (parameter-free, top-heavy)
        over S5(coherent-evidence) · S2(structure) · S3(content/symbol
        BM25) · S4(RM3) · S1(explicit-extraction). Content-BM25 spine
        guarantees totality ⇒ recall floor ≈ plain BM25.
        """
        qtok = _tok(issue)
        qsym: list[str] = []
        for m in _BACKTICK_RE.findall(issue):
            qsym += _split_ident(m)
        for ident in _IDENT_RE.findall(issue):
            qsym += _split_ident(ident)

        s5_soft, _ = self._s5(issue)
        s1 = self._s1(issue)
        s2 = self._s2(issue)
        s3_content = self._bm25_content.ranking(qtok)
        s3_symbol = self._bm25_symbol.ranking(qsym or qtok)
        s4_rm3 = self._bm25_content.ranking_weighted(self._rm3_query(qtok))

        fused = self._isr([s5_soft, s2, s3_content, s3_symbol,
                           s4_rm3, s1])
        ordered = sorted(
            fused.items(),
            key=lambda x: (-x[1], s3_content.index(x[0])
                           if x[0] in s3_content else 1 << 30),
        )
        out = [fid for fid, _ in ordered]
        seen = set(out)
        for fid in s3_content:                       # totality
            if fid not in seen:
                out.append(fid)
                seen.add(fid)
        return out[:k]

    def rerank(self, base_ranked: list[str], issue: str,
               k: int = 20) -> list[str]:
        """Zero-dep precision re-rank of a strong base ranking (e.g. the
        engine's, which already wins recall). ISR-fuse the base with the
        two precision signals embeddings can't replicate — S5 (exact
        contiguous code/error identity) and S2 (structure). This is the
        embeddings' own retrieve-then-rerank pattern, done zero-dep.

        Recall-safe by construction: the output is a PERMUTATION of the
        base candidates (no file dropped, none invented); the base's own
        ISR mass (1/r²) keeps confidently-ranked files near the top, so
        only strong, decorrelated S5/S2 agreement can promote a file —
        exactly the desired hit@5 / MRR lift without recall loss.
        """
        if not base_ranked:
            return self.rank(issue, k)
        s5_soft, _ = self._s5(issue)
        s2 = self._s2(issue)
        fused = self._isr([base_ranked, s5_soft, s2])
        pos = {f: i for i, f in enumerate(base_ranked)}
        ordered = sorted(
            (f for f in fused if f in pos),
            key=lambda f: (-fused[f], pos[f]),
        )
        seen = set(ordered)
        out = ordered + [f for f in base_ranked if f not in seen]
        return out[:k]

    # ── Personalized-PageRank rerank over the dependency graph ────────
    # The principled lever BM25 structurally cannot use: it treats files
    # as independent; PPR propagates a BM25 prior over the import graph
    # to its stationary distribution. r = (1-α)·p + α·Mᵀ·r, α=0.85
    # (the universal PageRank constant — NOT tuned on our data, so no
    # overfitting/artifact surface). Bounded to BM25's top-`pool`
    # candidates (won't surface garbage); whether net hit@10/@20 rises
    # is the falsifiable question — measured on a pre-registered eval,
    # NOT asserted.
    _PPR_ALPHA = 0.85
    _PPR_POOL = 50
    _PPR_ITERS = 60

    def rerank_ppr(self, issue: str, k: int = 20) -> list[str]:
        qw: dict[str, float] = defaultdict(float)
        for t in _tok(issue):
            qw[t] += 1.0
        scored = self._bm25_content.scores(qw)          # [(score, fid)] desc
        if not scored:
            return [fid for _, fid in scored][:k]
        pool = [fid for _, fid in scored[: self._PPR_POOL]]
        pool_set = set(pool)
        # Personalization prior p = L1-normalised BM25 over the pool.
        raw = {fid: max(0.0, s) for s, fid in scored[: self._PPR_POOL]}
        z = sum(raw.values()) or 1.0
        p = {f: raw[f] / z for f in pool}
        # Symmetrised import adjacency restricted to the pool.
        nbrs: dict[str, list[str]] = {f: [] for f in pool}
        for f in pool:
            for g in self.imports.get(f, ()):
                if g in pool_set:
                    nbrs[f].append(g)
            for g in self.imported_by.get(f, ()):
                if g in pool_set:
                    nbrs[f].append(g)
        a = self._PPR_ALPHA
        r = dict(p)
        for _ in range(self._PPR_ITERS):
            nxt = {f: (1.0 - a) * p[f] for f in pool}
            dangling = 0.0
            for f in pool:
                deg = len(nbrs[f])
                if deg == 0:
                    dangling += r[f]                    # teleport via p
                else:
                    share = a * r[f] / deg
                    for g in nbrs[f]:
                        nxt[g] += share
            if dangling:
                for f in pool:
                    nxt[f] += a * dangling * p[f]
            delta = sum(abs(nxt[f] - r[f]) for f in pool)
            r = nxt
            if delta < 1e-6:
                break
        bm_pos = {fid: i for i, (_, fid) in enumerate(scored)}
        ranked_pool = sorted(pool, key=lambda f: (-r[f], bm_pos[f]))
        out = ranked_pool + [fid for _, fid in scored
                             if fid not in pool_set]      # BM25 tail
        return out[:k]

    # ── engine_s6: deterministic edit-target rerank ───────────────────
    # NOT Pearl-style causal inference. A structural re-prior layered on
    # top of a strong base ranking (engine_s5), motivated by the n=36
    # forensic: 73% of misses were *source* files outranked by docs /
    # tests / non-source distractors; 27% had explicit cues we failed
    # to exploit. Four deterministic, non-tuned guards:
    #   • only re-prioritise inside the top-`_EDIT_WINDOW` of the base;
    #     the tail (rank > window) is preserved untouched → recall floor
    #   • explicit cues (S1 path/traceback + unique-symbol definers)
    #     are FROZEN at the top in cue-extraction order; never demoted
    #   • non-source files (rst/md/yml/cfg/toml/ini, docs/, .github/,
    #     HISTORY*, CHANGELOG*) are demoted within the window — UNLESS
    #     the issue carries doc/config intent (intent guard)
    #   • test files are demoted below source within the window —
    #     UNLESS the issue carries test intent (intent guard); when a
    #     test ranks in-window, its basename-mirrored non-test source
    #     file is promoted in just after it
    # Window=20 is the metric horizon (we report hit@5/10/20), not a
    # tuned threshold; everything else is a deterministic classifier.

    _EDIT_WINDOW = 20
    _NS_EXTS = (".rst", ".md", ".txt", ".yml", ".yaml", ".cfg",
                ".toml", ".ini")
    _NS_PATH_HINTS = ("docs/", "/doc/", ".github/", "/examples/",
                      "/tutorial/", "/tutorials/")
    _NS_NAME_HINTS = ("history", "changelog", "changes", "authors",
                      "contributors")
    _TEST_PATH_HINTS = ("/tests/", "tests/", "/test/")
    _DOC_INTENT_WORDS = (
        "documentation", "docstring", "tutorial", "readme", "rst",
        "the docs", "configuration file", " config file", "changelog",
        "yaml", " toml ", " ini ", "rst_prolog",
    )
    _TEST_INTENT_WORDS = (
        "unit test", "unit-test", "failing test", "test failure",
        "test case", "pytest fixture", "conftest", "broken test",
        "test that ", "regression test",
    )

    @staticmethod
    def _basename_noext(path: str) -> str:
        return path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    @classmethod
    def _is_test_path(cls, path: str) -> bool:
        p = path.lower()
        if any(h in p for h in cls._TEST_PATH_HINTS):
            return True
        base = p.rsplit("/", 1)[-1]
        return (base.startswith("test_") or base.endswith("_test.py")
                or base == "conftest.py")

    @classmethod
    def _is_non_source(cls, path: str) -> bool:
        p = path.lower()
        if any(p.endswith(e) for e in cls._NS_EXTS):
            return True
        if any(h in p for h in cls._NS_PATH_HINTS):
            return True
        base = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return any(h == base for h in cls._NS_NAME_HINTS)

    def _test_mirror(self, test_path: str) -> str | None:
        """basename(test_X.py) -> first non-test, non-non-source `X.py`
        anywhere in the corpus; deterministic, no scoring."""
        base = self._basename_noext(test_path).lower()
        if base.startswith("test_"):
            base = base[5:]
        elif base.endswith("_test"):
            base = base[:-5]
        if not base:
            return None
        target = "/" + base + ".py"
        for p in self.paths:
            pl = p.lower()
            if (pl.endswith(target) or pl == base + ".py") \
                    and not self._is_test_path(pl) \
                    and not self._is_non_source(pl):
                return p
        return None

    def rerank_edit_target(self, base_ranked: list[str], issue: str,
                           k: int = 20) -> list[str]:
        """Deterministic edit-target prior on top of `base_ranked`
        (intended: engine_s5). Window-local; tail preserved → recall
        floor; explicit cues frozen at top; class re-prio with intent
        guards; test→source mirror insertion."""
        if not base_ranked:
            return []
        window = base_ranked[: self._EDIT_WINDOW]
        tail = base_ranked[self._EDIT_WINDOW :]
        ilow = issue.lower()
        doc_intent = any(w in ilow for w in self._DOC_INTENT_WORDS)
        test_intent = any(w in ilow for w in self._TEST_INTENT_WORDS)

        # Frozen explicit cues (S1 + unique-symbol definers), in
        # extraction order; never altered later.
        cue_order: list[str] = []
        cue_set: set[str] = set()
        for p in self._s1(issue):
            if p in self._path_set and p not in cue_set:
                cue_order.append(p)
                cue_set.add(p)
        for s in self._issue_symbols(issue):
            fs = self.sym_def.get(s, ())
            if len(fs) == 1:
                f = next(iter(fs))
                if f not in cue_set:
                    cue_order.append(f)
                    cue_set.add(f)

        # Re-classify the remaining window. 0 = cue (frozen, handled
        # separately), 1 = source, 2 = test, 3 = non-source. Intent
        # guards collapse classes 2/3 to 1 selectively.
        rest = [f for f in window if f not in cue_set]
        orig_pos = {f: i for i, f in enumerate(rest)}

        def cls(f: str) -> int:
            if self._is_non_source(f):
                return 1 if doc_intent else 3
            if self._is_test_path(f):
                return 1 if test_intent else 2
            return 1

        rest_sorted = sorted(rest, key=lambda f: (cls(f), orig_pos[f]))

        # Test→source mirror: walk the re-prioritised window; when a
        # test appears, append its basename mirror (if any) immediately
        # after it. One mirror per test, never duplicated.
        built: list[str] = list(cue_order)
        seen: set[str] = set(cue_order)
        for f in rest_sorted:
            if f not in seen:
                built.append(f)
                seen.add(f)
            if self._is_test_path(f):
                m = self._test_mirror(f)
                if m and m not in seen:
                    built.append(m)
                    seen.add(m)

        # Totality: append the tail (engine_s5 order) for anything we
        # haven't placed yet. This is the recall floor — a relevant
        # file at engine rank 21 stays at engine rank 21 + cue_count.
        for f in tail:
            if f not in seen:
                built.append(f)
                seen.add(f)
        return built[:k]
