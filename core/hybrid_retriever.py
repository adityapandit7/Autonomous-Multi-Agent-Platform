from __future__ import annotations

"""
core/hybrid_retriever.py — Hybrid Retrieval (BM25 sparse + dense semantic + RRF fusion)

Replaces the original stub that concatenated symbolic and semantic results
without scoring, deduplication logic, or a build_symbol_index() function.

Research basis
--------------
- RRF (Cormack et al., SIGIR 2009): Reciprocal Rank Fusion fuses ranked lists
  from incompatible score scales by operating on *ranks*, not raw scores.
  Formula: score(d) = Σ 1/(k + rank_r(d)), k=60 is the standard constant.
  It is the zero-config default — no tuning required, robust across domains.

- Hybrid recall improvement (DEV Community, 2025): BM25 + dense hybrid
  consistently achieves 15-30% better recall than either method alone.
  BM25 catches exact identifiers (method names, class names, smell types);
  dense catches semantic matches ("refactor slow method" → "high complexity").

- Score-scale mismatch (Elasticsearch / OpenSearch, 2025): BM25 scores are
  unbounded; cosine similarity is in [-1, 1]. Never combine raw scores —
  always fuse via RRF which operates on rank position, not magnitude.

- Symbol boosting: results where a query token exactly matches the symbol
  name (simple name, last FQN segment) receive a rank boost before RRF,
  ensuring "OrderService.process" surfaces that exact function first.

Interface (matches pipeline.py and PlannerAgent imports exactly)
----------------------------------------------------------------
  build_symbol_index(parsed: dict) -> SymbolIndex
  hybrid_retrieve(query, retriever, symbol_index, top_k=10) -> list[dict]

Each returned dict has keys:
  symbol        str    fully-qualified name
  content       str    human-readable description
  score         float  RRF-fused relevance score (higher = more relevant)
  sources       list   which legs contributed: ["bm25", "semantic", "symbolic"]
  kind          str    "function" | "class"
  complexity    int    McCabe complexity (from parser)
  cohesion_score float LCOM cohesion score
  file_path     str    relative path within repo
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# RRF constant — k=60 from original Cormack et al. (SIGIR 2009).
_RRF_K = 60

_DEFAULT_TOP_K = 10
_RETRIEVER_TOP_K = 20   # fetch more from each leg before RRF re-ranks

# Exact-symbol-match score inflation before RRF.
# Treated as extra BM25 score so a direct identifier query always wins.
_EXACT_MATCH_BOOST = 5.0

# ---------------------------------------------------------------------------
# SymbolIndex
# ---------------------------------------------------------------------------

@dataclass
class SymbolIndex:
    """
    In-memory index of all symbols (functions + classes) from a parsed repo.

    symbols   List of symbol metadata dicts (one per function or class).
    corpus    Parallel list of BM25-indexable text strings.
    bm25      Fitted BM25Okapi instance, or None if rank_bm25 not installed.
    """
    symbols: list[dict[str, Any]] = field(default_factory=list)
    corpus: list[str] = field(default_factory=list)
    bm25: Any = None

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_STOP_TOKENS = {
    "a", "b", "c", "i", "s", "v", "n", "e",
    "get", "set", "is", "the", "of", "in", "to",
    "for", "and", "or", "this", "new",
}

def _tokenize(text: str) -> list[str]:
    """
    Lightweight tokenizer designed for Java code symbols.

    1. camelCase split  — "OrderService" → "Order Service"
    2. Lowercase + split on non-alphanumeric
    3. Remove single-char tokens and common Java noise words

    Without camelCase splitting, BM25 sees "OrderService" as one token
    and misses queries for "order" or "service" individually.
    """
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOP_TOKENS]

# ---------------------------------------------------------------------------
# Symbol document builder
# ---------------------------------------------------------------------------

def _symbol_to_document(sym: dict) -> str:
    """
    Build the BM25-indexable text representation for a symbol.

    Includes: name (camelCase-split), kind, docstring snippet, structural
    signals (high complexity, low cohesion), and top caller names.
    These match the tokens PlannerAgent injects into retrieval queries.
    """
    parts: list[str] = []

    name = sym.get("symbol", "")
    parts.append(name)
    parts.append(re.sub(r"([a-z])([A-Z])", r"\1 \2", name))
    parts.append(sym.get("kind", ""))

    content = sym.get("content", "")
    if content:
        parts.append(content[:200])

    if sym.get("complexity", 0) > 10:
        parts.append("high complexity")
    if sym.get("cohesion_score", 1.0) < 0.4:
        parts.append("low cohesion")
    for caller in sym.get("called_by", [])[:3]:
        parts.append(caller)

    return " ".join(str(p) for p in parts if p)

# ---------------------------------------------------------------------------
# build_symbol_index
# ---------------------------------------------------------------------------

def build_symbol_index(parsed: dict) -> SymbolIndex:
    """
    Convert JavaParser output into a SymbolIndex with fitted BM25.

    Called once in pipeline.py after Block 2 (parse), before PlannerAgent.

    Parameters
    ----------
    parsed  Dict from JavaParser.parse(), expected keys:
            "functions": list[dict], "classes": list[dict]

    Returns
    -------
    SymbolIndex with BM25 fitted on the symbol corpus.
    If rank_bm25 is not installed, BM25 leg is disabled (falls back to
    symbolic + semantic only).
    """
    symbols: list[dict[str, Any]] = []

    for fn in parsed.get("functions", []):
        name = fn.get("name", "unknown")
        params = fn.get("params", [])
        doc = fn.get("docstring") or ""
        symbols.append({
            "symbol":         name,
            "kind":           "function",
            "content": (
                f"{name}({', '.join(params)})"
                + (f" — {doc[:120]}" if doc else "")
            ),
            "complexity":     fn.get("complexity", 0),
            "loc":            fn.get("loc", 0),
            "cohesion_score": fn.get("cohesion_score", 1.0),
            "called_by":      fn.get("called_by", []),
            "file_path":      fn.get("file_path", ""),
        })

    for cls in parsed.get("classes", []):
        name = cls.get("name", "unknown")
        doc = cls.get("docstring") or ""
        methods = [m.get("name", "") for m in cls.get("methods", [])]
        symbols.append({
            "symbol":         name,
            "kind":           "class",
            "content": (
                f"class {name}"
                + (f" — {doc[:120]}" if doc else "")
                + (f" | methods: {', '.join(methods[:5])}" if methods else "")
            ),
            "complexity":     0,
            "loc":            cls.get("loc", 0),
            "cohesion_score": cls.get("cohesion_score", 1.0),
            "called_by":      [],
            "file_path":      cls.get("file_path", ""),
        })

    corpus = [_symbol_to_document(s) for s in symbols]

    bm25_index = None
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [_tokenize(doc) for doc in corpus]
        bm25_index = BM25Okapi(tokenized)
        logger.info("SymbolIndex built: %d symbol(s), BM25 fitted.", len(symbols))
    except ImportError:
        logger.warning(
            "rank_bm25 not installed — BM25 leg disabled. "
            "pip install rank-bm25  to enable it."
        )

    return SymbolIndex(symbols=symbols, corpus=corpus, bm25=bm25_index)

# ---------------------------------------------------------------------------
# Retrieval legs
# ---------------------------------------------------------------------------

def _bm25_retrieve(
    query: str,
    index: SymbolIndex,
    top_k: int,
) -> list[tuple[int, float]]:
    """
    BM25 sparse retrieval leg.

    Returns (symbol_idx, bm25_score) pairs sorted descending.
    Applies exact-symbol-match boost before sorting.
    Returns [] if BM25 is unavailable.

    Research: BM25 is the correct choice for exact identifier matches —
    class names, method names, and smell types that appear verbatim in
    the query.  Embedding models dilute rare code tokens across many
    dimensions and frequently miss exact identifiers (Premai, 2025).
    """
    if index.bm25 is None:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    raw_scores = index.bm25.get_scores(tokens)
    query_lower = query.lower()

    boosted: list[tuple[float, int]] = []
    for idx, score in enumerate(raw_scores):
        simple_name = index.symbols[idx].get("symbol", "").lower().split(".")[-1]
        if simple_name and simple_name in query_lower:
            score += _EXACT_MATCH_BOOST
        boosted.append((score, idx))

    boosted.sort(reverse=True)
    return [(idx, s) for s, idx in boosted[:top_k]]

def _symbolic_retrieve(
    query: str,
    index: SymbolIndex,
    top_k: int,
) -> list[tuple[int, float]]:
    """
    Exact symbolic retrieval leg — replaces the old symbolic_search import.

    Prefix match scores 2.0, substring match scores 1.0.
    This leg is fast (no ML), reliable for direct identifier lookups,
    and contributes its own RRF rank even when BM25 is disabled.
    """
    query_tokens = set(_tokenize(query))
    query_lower = query.lower()
    results: list[tuple[float, int]] = []

    for idx, sym in enumerate(index.symbols):
        name_lower = sym["symbol"].lower()
        simple_name = name_lower.split(".")[-1]
        score = 0.0

        if simple_name and any(simple_name.startswith(t) for t in query_tokens):
            score = 2.0
        elif any(t in name_lower for t in query_tokens):
            score = 1.0

        if score > 0:
            results.append((score, idx))

    results.sort(reverse=True)
    return [(idx, s) for s, idx in results[:top_k]]

def _semantic_retrieve(
    query: str,
    retriever,
    index: SymbolIndex,
    top_k: int,
) -> list[tuple[int, float]]:
    """
    Dense semantic retrieval leg via a LangChain-compatible retriever.

    Maps returned Document.metadata["symbol"] back to SymbolIndex positions
    for RRF.  Returns [] if retriever is None or raises.

    Research: Dense retrieval captures semantic similarity that exact-match
    methods miss — "refactor method with too many responsibilities" will match
    a function with a low cohesion score even without the exact phrase.
    """
    if retriever is None:
        return []

    try:
        docs = retriever.invoke(query)
    except Exception as exc:
        logger.warning("Semantic retriever failed (non-fatal): %s", exc)
        return []

    name_to_idx = {sym["symbol"]: i for i, sym in enumerate(index.symbols)}
    results: list[tuple[int, float]] = []

    for rank, doc in enumerate(docs[:top_k]):
        sym_name = doc.metadata.get("symbol", "")
        idx = name_to_idx.get(sym_name)
        if idx is not None:
            score = float(doc.metadata.get("score", 1.0 / (rank + 1)))
            results.append((idx, score))

    return results

# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    bm25_results: list[tuple[int, float]],
    semantic_results: list[tuple[int, float]],
    symbolic_results: list[tuple[int, float]],
    index: SymbolIndex,
    top_k: int,
    k: int = _RRF_K,
) -> list[dict[str, Any]]:
    """
    Fuse three ranked lists via Reciprocal Rank Fusion.

    RRF formula:  score(d) = Σ_r  1 / (k + rank_r(d))

    A symbol found by all three legs ranks higher than one found by only one.
    k=60 is the original paper constant — larger k flattens score differences,
    smaller k creates steeper differences between top ranks.

    Research (OpenSearch 2025): RRF is preferred over min-max or L2
    normalization because it handles incompatible score scales (BM25 is
    unbounded; cosine similarity is bounded) without any tuning.
    """
    rrf_scores: dict[int, float] = {}
    sources: dict[int, list[str]] = {}

    legs = [
        (bm25_results,     "bm25"),
        (semantic_results, "semantic"),
        (symbolic_results, "symbolic"),
    ]

    for ranked_list, leg_name in legs:
        for rank, (sym_idx, _) in enumerate(ranked_list):
            rrf_scores[sym_idx] = rrf_scores.get(sym_idx, 0.0) + 1.0 / (k + rank + 1)
            sources.setdefault(sym_idx, []).append(leg_name)

    ordered = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)

    results: list[dict[str, Any]] = []
    for sym_idx in ordered[:top_k]:
        sym = index.symbols[sym_idx]
        results.append({
            "symbol":         sym["symbol"],
            "content":        sym["content"],
            "score":          round(rrf_scores[sym_idx], 6),
            "sources":        sorted(set(sources[sym_idx])),
            "kind":           sym.get("kind", ""),
            "complexity":     sym.get("complexity", 0),
            "cohesion_score": sym.get("cohesion_score", 1.0),
            "file_path":      sym.get("file_path", ""),
        })

    return results

# ---------------------------------------------------------------------------
# hybrid_retrieve — public entry point
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    query: str,
    retriever,
    symbol_index: SymbolIndex,
    top_k: int = _DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Run BM25 + symbolic + semantic retrieval and fuse results with RRF.

    Parameters
    ----------
    query           Retrieval query built by PlannerAgent._build_retrieval_query().
    retriever       LangChain-compatible retriever (invoke(query) → [Document]).
                    Pass None to skip the semantic leg (e.g. in unit tests).
    symbol_index    SymbolIndex returned by build_symbol_index(parsed).
    top_k           Number of final results to return. Default 10.

    Returns
    -------
    List of result dicts sorted by RRF score descending, max length top_k.

    Leg summary
    -----------
    BM25 leg        rank_bm25.BM25Okapi on camelCase-split symbol corpus.
                    Disabled if rank_bm25 not installed.
                    Best for: exact names, smell types ("LongMethod"),
                              class/method identifiers.

    Symbolic leg    Prefix + substring match on symbol names.
                    Always available, no dependencies.
                    Best for: direct identifier queries ("OrderService").

    Semantic leg    retriever.invoke(query) — dense embedding search.
                    Disabled if retriever is None.
                    Best for: conceptual queries ("high complexity",
                              "low cohesion function", "refactor candidate").

    RRF fusion      Combines all three legs. A symbol appearing in multiple
                    legs ranks higher than one found by a single leg.
                    k=60 standard constant, no tuning needed.
    """
    if not symbol_index.symbols:
        logger.warning("hybrid_retrieve called with empty SymbolIndex — returning [].")
        return []

    fetch_k = min(_RETRIEVER_TOP_K, len(symbol_index.symbols))

    bm25_results     = _bm25_retrieve(query, symbol_index, fetch_k)
    symbolic_results = _symbolic_retrieve(query, symbol_index, fetch_k)
    semantic_results = _semantic_retrieve(query, retriever, symbol_index, fetch_k)

    logger.debug(
        "Retrieval legs — bm25: %d  symbolic: %d  semantic: %d",
        len(bm25_results), len(symbolic_results), len(semantic_results),
    )

    fused = _rrf_fuse(
        bm25_results,
        semantic_results,
        symbolic_results,
        index=symbol_index,
        top_k=top_k,
        k=_RRF_K,
    )

    logger.debug("hybrid_retrieve returning %d result(s) after RRF fusion.", len(fused))
    return fused