# scripts/query_rag.py
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np

# ============================================================
# Suppress TF/oneDNN logs 
# ============================================================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all,1=info,2=warn,3=error
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # reduce oneDNN noise


# ============================================================
# Ensure project root is on sys.path so `import backend...` works
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse your project embedder (same model used during indexing)
from backend.api.rag.embedder import embed_texts  # noqa: E402

INDEX_DIR = ROOT / "rag_index"


def _find_first(patterns: List[str]) -> Path:
    for pat in patterns:
        hits = list(INDEX_DIR.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not find any of: {patterns} inside {INDEX_DIR}")


def _load_chunks(path: Path) -> List[Dict[str, Any]]:
    """
    Supports:
    - JSON list
    - JSONL
    - JSON dict with key "chunks"
    """
    text = path.read_text(encoding="utf-8")
    text_stripped = text.lstrip()

    if text_stripped.startswith("{"):
        obj = json.loads(text)
        if isinstance(obj, dict) and "chunks" in obj and isinstance(obj["chunks"], list):
            return obj["chunks"]
        raise ValueError(f"Unsupported JSON dict format in {path} (expected key 'chunks').")

    if text_stripped.startswith("["):
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        raise ValueError(f"Unsupported JSON list format in {path}.")

    # assume JSONL
    chunks: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        chunks.append(json.loads(line))
    return chunks


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python scripts/query_rag.py "your question" [top_k]')
        return 2

    question = sys.argv[1]
    top_k = _safe_int(sys.argv[2] if len(sys.argv) >= 3 else None, default=5)
    if top_k < 1:
        top_k = 1
    if top_k > 20:
        top_k = 20

    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"rag_index folder not found at: {INDEX_DIR}")

    # Try common filenames
    index_path = _find_first(["*.faiss", "*.index", "index.faiss", "faiss.index"])
    chunks_path = _find_first(["chunks.json", "chunks.jsonl", "metadata.json", "documents.json"])

    print(f"[INFO] Using index:  {index_path.name}")
    print(f"[INFO] Using chunks: {chunks_path.name}")
    print(f"[INFO] top_k={top_k}")

    index = faiss.read_index(str(index_path))
    chunks = _load_chunks(chunks_path)

    qvec = embed_texts([question])  # expected: (1, d)
    if isinstance(qvec, list):
        qvec = np.array(qvec, dtype=np.float32)
    else:
        qvec = np.asarray(qvec, dtype=np.float32)

    if qvec.ndim != 2 or qvec.shape[0] != 1:
        raise ValueError(f"embed_texts output must be 2D with shape (1, d). Got shape: {qvec.shape}")

    distances, indices = index.search(qvec, top_k)

    print("\n=== QUERY ===")
    print(question)
    print("\n=== TOP MATCHES ===")

    for rank, (i, d) in enumerate(zip(indices[0], distances[0]), start=1):
        if i < 0 or i >= len(chunks):
            print(f"\n[{rank}] idx={i} dist={float(d):.4f} (OUT OF RANGE for chunks)")
            continue

        ch = chunks[i]
        text = ch.get("text") or ch.get("chunk") or ch.get("content") or ""
        source = ch.get("source") or ch.get("file") or ch.get("pdf") or "unknown"
        page = ch.get("page", ch.get("page_number", None))
        chunk_id = ch.get("chunk_id", ch.get("id", i))

        print(f"\n[{rank}] idx={i} dist={float(d):.4f} | source={source} | page={page} | chunk_id={chunk_id}")
        preview = text.strip()
        print(preview[:800] + ("..." if len(preview) > 800 else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
