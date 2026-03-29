from __future__ import annotations

"""
core/rag_query.py — RAG Query Path

Position in pipeline (architecture diagram — bottom half)
-----------------------------------------------------------
  Update Documentation
    ↓
  Embedding → Retriever ← User Query
                ↓
          Context + Query
                ↓
          LLM Generator
                ↓
          Final Output (documentation / refactoring answer)

This module implements the user-facing query path — separate from the
autonomous agent pipeline that runs on ingestion.  Users ask natural-language
questions about the repo ("Why was OrderService refactored?", "What does
processOrder do?", "Which methods have high complexity?") and get grounded,
context-aware answers.

Research basis
--------------
- Context Engineering (RAGFlow 2025): the shift from optimising single
  retrieval algorithms to the systematic design of the end-to-end
  retrieval → context assembly → model reasoning pipeline.  Context quality
  determines answer quality more than model size.

- Retrieval-first, long-context containment (RAGFlow 2025): retrieve with
  small, precise chunks (256–512 tokens) for high recall; assemble a larger
  context window for generation.  Do NOT stuff the full codebase into the
  prompt — "Lost in the Middle" degrades answers non-linearly.

- Prompt engineering for RAG (Stack Overflow blog 2024): the prompt must
  (a) include the retrieved context, (b) format it with clear structural
  markers so the LLM can distinguish instructions from data, and
  (c) instruct the model to stay grounded in the provided context.

- Smell-aware context boosting (project-specific): results tagged with
  has_smell=True and high severity_weight should appear first in the
  assembled context — they are the highest-value chunks for a refactoring
  assistant.

- RAGOps observability (arXiv 2506.03401, 2025): log all queries, retrieved
  chunks, and generated responses for evaluation and continuous improvement.

Interface
---------
  query_repo(
      query: str,
      retriever,
      symbol_index: SymbolIndex,
      llm,
      *,
      top_k: int = 8,
      mode: str = "answer",   # "answer" | "explain" | "smell"
  ) -> RAGResponse
"""

import logging
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

from core.hybrid_retriever import hybrid_retrieve, SymbolIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token-budget constants
# (1 token ≈ 4 chars; CodeT5+ max_input is typically 512 tokens = 2048 chars)
# ---------------------------------------------------------------------------
_MAX_CONTEXT_CHARS = 3_000   # total context block fed to the LLM
_MAX_CHUNK_CHARS   = 600     # max chars per individual retrieved chunk

# Query modes → system instruction snippets
_MODE_INSTRUCTIONS: dict[str, str] = {
    "answer": (
        "You are a Java code assistant. Answer the question using ONLY the "
        "context provided. Be concise and precise. If the context does not "
        "contain enough information to answer, say so explicitly."
    ),
    "explain": (
        "You are a Java code explainer. Using ONLY the context provided, "
        "explain the purpose, behaviour, and design of the requested code "
        "element. Include relevant metrics if they help the explanation."
    ),
    "smell": (
        "You are a Java code quality analyst. Using ONLY the context provided, "
        "describe the detected code smells, their severity, and the recommended "
        "refactoring approach. Prioritise critical and high-severity smells."
    ),
}

# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    """
    Structured output of query_repo().

    Fields
    ------
    query           The original user query.
    answer          Generated answer text.
    sources         List of source symbol names used in context.
    context_chars   Total chars of context fed to the LLM (for token accounting).
    retrieval_ms    Retrieval wall-clock time in milliseconds.
    generation_ms   Generation wall-clock time in milliseconds.
    mode            Query mode used.
    chunks          Full list of retrieved chunk dicts (for RAGOps logging).
    """
    query:          str
    answer:         str
    sources:        list[str]       = field(default_factory=list)
    context_chars:  int             = 0
    retrieval_ms:   float           = 0.0
    generation_ms:  float           = 0.0
    mode:           str             = "answer"
    chunks:         list[dict]      = field(default_factory=list)

# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _assemble_context(chunks: list[dict], max_chars: int) -> tuple[str, list[str]]:
    """
    Assemble retrieved chunks into a single context block.

    Ordering strategy (research: smell-aware boosting + diversity):
    1. Chunks with has_smell=True ordered by severity_weight descending
       (critical → high → medium → low)
    2. Remaining chunks ordered by RRF score descending
    3. Truncate at max_chars so we stay within the LLM's token budget

    Each chunk is wrapped with a clear structural marker so the LLM can
    distinguish instructions from data (LangChain RAG tutorial, 2025).

    Returns
    -------
    (context_block, sources)
        context_block  Formatted string ready to inject into the prompt.
        sources        List of symbol names used (for RAGResponse.sources).
    """
    # Sort: smell chunks first by severity weight, then rest by RRF score
    smell_chunks = sorted(
        [c for c in chunks if c.get("has_smell") or c.get("kind") == "smell"],
        key=lambda c: c.get("severity_weight", 0.0),
        reverse=True,
    )
    other_chunks = sorted(
        [c for c in chunks if not c.get("has_smell") and c.get("kind") != "smell"],
        key=lambda c: c.get("score", 0.0),
        reverse=True,
    )
    ordered = smell_chunks + other_chunks

    parts:   list[str] = []
    sources: list[str] = []
    total   = 0

    for i, chunk in enumerate(ordered):
        symbol  = chunk.get("symbol", "unknown")
        content = chunk.get("content", "")
        kind    = chunk.get("kind", "")
        score   = chunk.get("score", 0.0)

        # Annotate with smell severity if present
        smell_note = ""
        if chunk.get("has_smell"):
            smell_note = (
                f"  [SMELL: {', '.join(chunk.get('smell_types', []))} "
                f"| severity: {chunk.get('max_severity', '?')}]"
            )

        header = f"[{i+1}] {kind.upper()}: {symbol}{smell_note}  (score: {score:.4f})"
        body   = textwrap.shorten(content, width=_MAX_CHUNK_CHARS, placeholder="…")
        block  = f"{header}\n{body}"

        if total + len(block) > max_chars:
            break

        parts.append(block)
        sources.append(symbol)
        total += len(block)

    context_block = "\n\n---\n\n".join(parts)
    return context_block, sources

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(query: str, context: str, mode: str) -> str:
    """
    Build the full prompt for the LLM generator.

    Structure (research: Stack Overflow blog 2024 / LangChain RAG docs):
    1. System instruction (mode-specific, grounding directive)
    2. <context> XML block — clear delimiter so the model separates
       retrieved data from instructions (protects against prompt injection)
    3. Question

    XML-style delimiters are recommended by LangChain (2025) for RAG prompts
    because they reduce the risk of the model treating context as instructions.
    """
    instruction = _MODE_INSTRUCTIONS.get(mode, _MODE_INSTRUCTIONS["answer"])

    return textwrap.dedent(f"""
        {instruction}

        <context>
        {context}
        </context>

        Question: {query}

        Answer:
    """).strip()

# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

def _generate(prompt: str, llm) -> str:
    """
    Call the LLM generator and return the answer string.

    Supports three LLM interfaces in priority order:
    1. invoke(prompt)  — LangChain LCEL interface (returns AIMessage or str)
    2. __call__(prompt)— legacy LangChain interface
    3. generate(prompt)— custom pipeline interface (returns dict with "text")

    Falls back gracefully to a placeholder if generation fails.
    """
    try:
        if hasattr(llm, "invoke"):
            result = llm.invoke(prompt)
            # LangChain AIMessage has .content; plain str is also valid
            return result.content if hasattr(result, "content") else str(result)

        if callable(llm):
            result = llm(prompt)
            return result.get("text", str(result)) if isinstance(result, dict) else str(result)

    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        return f"[Generation error: {exc}]"

    return "[No LLM interface available]"

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def query_repo(
    query: str,
    retriever,
    symbol_index: SymbolIndex,
    llm,
    *,
    top_k: int = 8,
    mode: str = "answer",
) -> RAGResponse:
    """
    Answer a natural-language question about the repo using RAG.

    Flow
    ----
    1. Hybrid retrieve (BM25 + symbolic + semantic) top_k chunks
    2. Smell-aware context assembly with token budget
    3. Prompt construction with XML delimiters
    4. LLM generation
    5. Return structured RAGResponse

    Parameters
    ----------
    query           Natural-language question from the user.
    retriever       LangChain-compatible retriever (or None for BM25+symbolic only).
    symbol_index    SymbolIndex from build_symbol_index(parsed).
    llm             LLM instance with invoke() or __call__() interface.
    top_k           Number of chunks to retrieve (default 8).
    mode            "answer"  — factual Q&A about the repo
                    "explain" — detailed explanation of a code element
                    "smell"   — code quality and refactoring guidance

    Returns
    -------
    RAGResponse with answer, sources, timing, and full chunk list.
    """
    if not query.strip():
        return RAGResponse(query=query, answer="[Empty query.]", mode=mode)

    # ── 1. Retrieval ──────────────────────────────────────────────────────
    t0 = time.monotonic()
    chunks = hybrid_retrieve(query, retriever, symbol_index, top_k=top_k)
    retrieval_ms = (time.monotonic() - t0) * 1000

    if not chunks:
        logger.warning("query_repo: no chunks retrieved for query: %s", query)
        return RAGResponse(
            query=query,
            answer="No relevant code context found for this query.",
            mode=mode,
            retrieval_ms=retrieval_ms,
        )

    logger.info(
        "query_repo: retrieved %d chunk(s) in %.1fms  mode=%s",
        len(chunks), retrieval_ms, mode,
    )

    # ── 2. Context assembly ───────────────────────────────────────────────
    context_block, sources = _assemble_context(chunks, _MAX_CONTEXT_CHARS)

    # ── 3. Prompt construction ────────────────────────────────────────────
    prompt = _build_prompt(query, context_block, mode)

    logger.debug(
        "query_repo: prompt length %d chars  context %d chars  sources: %s",
        len(prompt), len(context_block), sources,
    )

    # ── 4. LLM generation ─────────────────────────────────────────────────
    t1 = time.monotonic()
    answer = _generate(prompt, llm)
    generation_ms = (time.monotonic() - t1) * 1000

    logger.info("query_repo: generated answer in %.1fms", generation_ms)

    # ── 5. RAGOps logging (query + sources + timing) ──────────────────────
    # Research (RAGOps, arXiv 2025): log all queries and retrieved chunks
    # for evaluation pipelines (nDCG, LLM-as-judge) and continuous improvement.
    logger.debug(
        "RAGOps | query=%r | sources=%s | retrieval_ms=%.1f | generation_ms=%.1f",
        query, sources, retrieval_ms, generation_ms,
    )

    return RAGResponse(
        query=query,
        answer=answer,
        sources=sources,
        context_chars=len(context_block),
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        mode=mode,
        chunks=chunks,
    )