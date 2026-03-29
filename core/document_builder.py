from __future__ import annotations

"""
core/document_builder.py — Document Builder for RAG Indexing

Position in pipeline (architecture diagram)
--------------------------------------------
  Semantic Units (JavaParser output + RepoState)
    ↓
  → [THIS FILE] Document Builder
      - builds LangChain Document objects with rich metadata
      - AST-boundary chunking (one logical unit per document)
      - smell-awareness: smell type + severity injected into metadata
    ↓
  Embedding / Index
    ↓
  Knowledge Base (vector store queried by hybrid_retriever.py)

Research basis
--------------
- cAST (ACL Findings, Nov 2025 / arXiv 2506.15655): AST-boundary chunking
  yields self-contained, syntactically complete chunks that improve retrieval
  accuracy by up to 5.5 points over fixed-size chunking.  Function and class
  boundaries are the natural AST unit for Java code — never split across them.

- Metadata enrichment (Elasticsearch Labs, 2024; Databricks, 2025):
  enriching chunks with structural signals (complexity, cohesion, smell_type,
  severity) enables filtered retrieval and composite embeddings that outperform
  content-only chunks.

- LangChain Document contract (LangChain docs, 2025): the semantic retriever
  leg in hybrid_retriever.py calls doc.metadata.get("symbol") — documents
  MUST be LangChain Document objects, not plain dicts.

- Chunk size (NVIDIA 2024 benchmark; Chroma Research 2024): factoid/analytical
  queries over code perform best at 256–512 tokens. We use a character-based
  proxy (1 token ≈ 4 chars) with a configurable MAX_CHARS_PER_CHUNK.

- Parent-chunk context (HuggingFace cookbook, 2025): the LLM needs the method
  body for refactoring/documentation; the retriever needs a compact summary
  for precise embedding. We write a short "retrieval chunk" (signature +
  docstring + metrics) and a longer "generation chunk" (body snippet) as
  separate documents with a shared parent_id so they can be joined later.
"""

import hashlib
import logging
import textwrap
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunk-size constants
# (1 token ≈ 4 chars; target 400 tokens ≈ 1600 chars for retrieval chunks)
# ---------------------------------------------------------------------------
MAX_RETRIEVAL_CHARS = 1_600    # retrieval chunk — compact, precise embedding
MAX_BODY_CHARS      = 4_000    # generation chunk — full context for LLM

# Severity → retrieval weight (used as a float metadata field for boosted search)
_SEVERITY_WEIGHT = {
    "critical": 1.0,
    "high":     0.75,
    "medium":   0.5,
    "low":      0.25,
}

# ---------------------------------------------------------------------------
# Try to import LangChain Document.
# Fall back to a minimal shim so the module loads even without langchain.
# ---------------------------------------------------------------------------
try:
    from langchain_core.documents import Document as LangchainDocument
    _HAS_LANGCHAIN = True
except ImportError:
    try:
        from langchain.docstore.document import Document as LangchainDocument
        _HAS_LANGCHAIN = True
    except ImportError:
        logger.warning(
            "langchain not installed — build_documents returns plain dicts. "
            "pip install langchain-core  to enable LangChain Document output."
        )
        _HAS_LANGCHAIN = False

        class LangchainDocument:          # type: ignore[no-redef]
            """Minimal shim used when langchain is not installed."""
            def __init__(self, page_content: str, metadata: dict):
                self.page_content = page_content
                self.metadata = metadata

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stable_id(*parts: str) -> str:
    """Deterministic document ID from symbol name + chunk type."""
    return hashlib.md5(":".join(parts).encode()).hexdigest()[:12]

def _smells_for(symbol_name: str, repo_state) -> list[dict]:
    """Return all CodeSmell objects that target a given symbol name."""
    return [
        s for s in repo_state.smells
        if s.location == symbol_name
    ]

def _smell_metadata(smells: list) -> dict[str, Any]:
    """
    Distil a list of CodeSmell objects into flat metadata fields.

    Stores the highest-severity smell type and weight so vector stores
    that support metadata filtering can filter by severity, and so
    composite embeddings can weight smell-bearing chunks higher.
    """
    if not smells:
        return {
            "has_smell":       False,
            "smell_types":     [],
            "max_severity":    "none",
            "severity_weight": 0.0,
        }

    # Pick the most critical smell as the primary signal
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
    top = max(smells, key=lambda s: severity_order.get(s.severity.name.lower(), 0))

    return {
        "has_smell":       True,
        "smell_types":     list({s.smell_type for s in smells}),
        "max_severity":    top.severity.name.lower(),
        "severity_weight": _SEVERITY_WEIGHT.get(top.severity.name.lower(), 0.0),
        "smell_count":     len(smells),
        "smell_reasoning": top.reasoning[:200] if top.reasoning else "",
    }

def _truncate(text: str, max_chars: int, label: str = "") -> str:
    """Truncate text to max_chars with an ellipsis note."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    note = f"\n... [{label} truncated at {max_chars} chars]" if label else "\n..."
    return cut + note

# ---------------------------------------------------------------------------
# Per-symbol document builders
# ---------------------------------------------------------------------------

def _function_documents(fn, repo_state) -> list[LangchainDocument]:
    """
    Build 1–2 LangChain Documents for a single FunctionUnit.

    Document 1 — retrieval chunk (always emitted)
      Compact: signature + docstring + structural metrics + smell summary.
      Targeted at the embedding model: fits in 400 tokens, one clear topic.

    Document 2 — generation chunk (emitted only if raw_code available)
      The actual method body from raw_code, bounded to MAX_BODY_CHARS.
      Targeted at the LLM generator: gives full context for refactoring/docs.
      Shares parent_id with the retrieval chunk for join queries.

    Research: parent-chunk / child-chunk pattern (HuggingFace cookbook, 2025)
    — retrieve with small, precise chunks; generate with large, contextual ones.
    """
    smells = _smells_for(fn.name, repo_state)
    smell_meta = _smell_metadata(smells)

    param_str = ", ".join(fn.params) if fn.params else "none"
    caller_str = ", ".join(fn.called_by[:5]) if fn.called_by else "none"
    callee_str = ", ".join(fn.calls[:5]) if fn.calls else "none"
    docstring  = fn.docstring or "No docstring."

    # ── Retrieval chunk ───────────────────────────────────────────────────
    retrieval_text = textwrap.dedent(f"""
        Function: {fn.name}
        Parameters: {param_str}
        File: {fn.file_path or 'unknown'}  Lines: {fn.start_line}–{fn.end_line}

        Docstring:
        {docstring}

        Structural metrics:
          Complexity: {fn.complexity}  LOC: {fn.loc}  Cohesion: {fn.cohesion_score:.2f}

        Dependencies:
          Calls: {callee_str}
          Called by: {caller_str}
    """).strip()

    if smell_meta["has_smell"]:
        smell_line = (
            f"\nCode smells: {', '.join(smell_meta['smell_types'])} "
            f"[{smell_meta['max_severity']}]"
            + (f"\n  Reasoning: {smell_meta['smell_reasoning']}" if smell_meta["smell_reasoning"] else "")
        )
        retrieval_text += smell_line

    retrieval_text = _truncate(retrieval_text, MAX_RETRIEVAL_CHARS, "retrieval")

    parent_id = _stable_id(fn.name, "function")

    base_meta: dict[str, Any] = {
        # --- identity ---
        "id":          parent_id,
        "symbol":      fn.name,
        "kind":        "function",
        "chunk_type":  "retrieval",
        "file_path":   fn.file_path,
        "start_line":  fn.start_line,
        "end_line":    fn.end_line,
        # --- structural ---
        "complexity":        fn.complexity,
        "loc":               fn.loc,
        "cohesion_score":    fn.cohesion_score,
        "param_count":       len(fn.params),
        "caller_count":      len(fn.called_by),
        # --- smell signals ---
        **smell_meta,
    }

    docs = [LangchainDocument(page_content=retrieval_text, metadata=base_meta)]

    # ── Generation chunk (body snippet from raw_code) ─────────────────────
    body = _extract_body(fn, repo_state.raw_code)
    if body:
        gen_text = _truncate(
            f"// {fn.name}\n{body}", MAX_BODY_CHARS, "body"
        )
        gen_meta = {**base_meta, "chunk_type": "generation", "parent_id": parent_id}
        gen_meta["id"] = _stable_id(fn.name, "function", "generation")
        docs.append(LangchainDocument(page_content=gen_text, metadata=gen_meta))

    return docs

def _class_documents(cls, repo_state) -> list[LangchainDocument]:
    """
    Build 1 LangChain Document for a ClassUnit.

    Classes are indexed at the class level (not method level) because:
    1. Class-level smells (GodClass, LargeClass) target the class.
    2. Method-level documents are already indexed by _function_documents.
    3. cAST research: class boundary is a valid AST chunk unit.
    """
    smells  = _smells_for(cls.name, repo_state)
    smell_meta = _smell_metadata(smells)

    methods     = [m.name for m in cls.methods] if cls.methods else []
    method_str  = ", ".join(methods[:10]) + ("..." if len(methods) > 10 else "")
    docstring   = cls.docstring or "No docstring."

    content = textwrap.dedent(f"""
        Class: {cls.name}
        File: {cls.file_path or 'unknown'}
        Abstract: {cls.is_abstract}
        Superclass: {cls.superclass or 'none'}
        Interfaces: {', '.join(cls.interfaces) if cls.interfaces else 'none'}

        Docstring:
        {docstring}

        Methods ({len(methods)}):
        {method_str}

        Structural metrics:
          LCOM: {cls.lcom:.2f}  Instability: {cls.instability:.2f}
    """).strip()

    if smell_meta["has_smell"]:
        content += (
            f"\nCode smells: {', '.join(smell_meta['smell_types'])} "
            f"[{smell_meta['max_severity']}]"
        )

    content = _truncate(content, MAX_RETRIEVAL_CHARS, "class")

    meta: dict[str, Any] = {
        "id":              _stable_id(cls.name, "class"),
        "symbol":          cls.name,
        "kind":            "class",
        "chunk_type":      "retrieval",
        "file_path":       cls.file_path,
        "method_count":    len(methods),
        "is_abstract":     cls.is_abstract,
        "lcom":            cls.lcom,
        "instability":     cls.instability,
        **smell_meta,
    }

    return [LangchainDocument(page_content=content, metadata=meta)]

def _smell_documents(repo_state) -> list[LangchainDocument]:
    """
    Emit one Document per unique (smell_type, location) pair.

    Research (Elasticsearch Labs, 2024): separate smell documents enable
    targeted filtered retrieval by smell type or severity without polluting
    the function/class documents.  The Planner can query
    "LongMethod HIGH severity" and retrieve specifically smell documents.
    """
    docs: list[LangchainDocument] = []

    for smell in repo_state.smells:
        content = textwrap.dedent(f"""
            Code smell: {smell.smell_type}
            Target: {smell.location}
            Severity: {smell.severity.name}
            Confidence: {smell.confidence:.2f}
            Description: {smell.description}
            Reasoning: {smell.reasoning or 'none'}
        """).strip()

        meta: dict[str, Any] = {
            "id":            _stable_id(smell.smell_type, smell.location),
            "symbol":        smell.location,
            "kind":          "smell",
            "chunk_type":    "smell",
            "smell_type":    smell.smell_type,
            "max_severity":  smell.severity.name.lower(),
            "severity_weight": _SEVERITY_WEIGHT.get(smell.severity.name.lower(), 0.0),
            "confidence":    smell.confidence,
            "agent_id":      smell.agent_id,
        }

        docs.append(LangchainDocument(page_content=content, metadata=meta))

    return docs

# ---------------------------------------------------------------------------
# Body extraction helper
# ---------------------------------------------------------------------------

def _extract_body(fn, raw_code: str) -> str:
    """
    Extract the method body from raw_code using start/end line numbers.

    Returns empty string if line numbers are unavailable or out of range.
    The body is used for the generation chunk only — not the retrieval chunk.
    """
    if not raw_code or not fn.start_line or not fn.end_line:
        return ""

    lines = raw_code.splitlines()
    start = max(0, fn.start_line - 1)
    end   = min(len(lines), fn.end_line)

    if start >= end:
        return ""

    return "\n".join(lines[start:end])

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_documents(repo_state) -> list[LangchainDocument]:
    """
    Build all LangChain Documents from a RepoState for RAG indexing.

    Called by document_builder after parsing + RepoState construction.
    The returned documents are ready to embed and upsert to a vector store.

    Document types emitted
    ----------------------
    retrieval chunk   (function)  signature + metrics + smell summary
    generation chunk  (function)  raw body snippet, linked by parent_id
    retrieval chunk   (class)     class overview + metrics + smell summary
    smell chunk                   one document per smell instance

    Parameters
    ----------
    repo_state  A RepoState instance (from state.py).

    Returns
    -------
    List of LangchainDocument objects (or plain-dict shims if langchain
    is not installed).
    """
    documents: list[LangchainDocument] = []

    for fn in repo_state.functions:
        try:
            documents.extend(_function_documents(fn, repo_state))
        except Exception as exc:
            logger.warning("Failed to build documents for function %s: %s", fn.name, exc)

    for cls in repo_state.classes:
        try:
            documents.extend(_class_documents(cls, repo_state))
        except Exception as exc:
            logger.warning("Failed to build documents for class %s: %s", cls.name, exc)

    if repo_state.smells:
        try:
            documents.extend(_smell_documents(repo_state))
        except Exception as exc:
            logger.warning("Failed to build smell documents: %s", exc)

    logger.info(
        "build_documents: %d document(s) built "
        "(%d functions, %d classes, %d smells).",
        len(documents),
        len(repo_state.functions),
        len(repo_state.classes),
        len(repo_state.smells),
    )

    return documents

def upsert_to_vectorstore(
    documents: list[LangchainDocument],
    vectorstore,
    *,
    batch_size: int = 50,
) -> int:
    """
    Embed and upsert documents into a LangChain-compatible vector store.

    Uses the document's metadata["id"] as the vector store ID so re-runs
    produce deterministic IDs and avoid duplicate embeddings for the same
    symbol — the vector store's upsert semantics update existing documents
    rather than appending duplicates.

    Parameters
    ----------
    documents     List from build_documents().
    vectorstore   Any LangChain VectorStore (Chroma, FAISS, Pinecone, etc.)
                  that implements add_documents(docs, ids=[...]).
    batch_size    Number of documents per upsert batch.
                  Default 50 avoids embedding API rate limits.

    Returns
    -------
    Total number of documents upserted.
    """
    if not documents:
        logger.warning("upsert_to_vectorstore: no documents to upsert.")
        return 0

    total = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        ids   = [doc.metadata.get("id", _stable_id(str(i + j))) for j, doc in enumerate(batch)]

        try:
            vectorstore.add_documents(batch, ids=ids)
            total += len(batch)
            logger.debug("Upserted batch %d–%d (%d docs).", i, i + len(batch), len(batch))
        except Exception as exc:
            logger.error("upsert_to_vectorstore batch %d failed: %s", i, exc)

    logger.info("upsert_to_vectorstore: %d/%d document(s) upserted.", total, len(documents))
    return total