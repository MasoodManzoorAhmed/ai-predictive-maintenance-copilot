from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Tuple

import faiss  # type: ignore
import numpy as np
from pypdf import PdfReader  # type: ignore

from backend.api.rag.docstore import Chunk, DocStore
from backend.api.rag.embedder import embed_texts


# -------------------------
# Text cleaning / chunking
# -------------------------
def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def chunk_text(text: str, chunk_chars: int = 900, overlap: int = 150) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    out: List[str] = []
    i = 0
    step = max(1, chunk_chars - overlap)
    while i < len(text):
        out.append(text[i : i + chunk_chars])
        i += step
    return out


def read_pdf(path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: List[Tuple[int, str]] = []
    for i, p in enumerate(reader.pages, start=1):
        t = p.extract_text() or ""
        t = clean_text(t)
        if t:
            pages.append((i, t))
    return pages


# -------------------------
# Helpers
# -------------------------
def stable_doc_id(pdf_path: Path) -> str:
    """
    Create a stable doc_id that avoids collisions when filenames repeat.
    Example: compressor_rotating_stall_surge_modeling_control__a1b2c3d4
    """
    name = pdf_path.stem.strip().lower()
    h = hashlib.md5(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{name}__{h}"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalization is important if you're using IndexFlatIP (cosine similarity style).
    """
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


# -------------------------
# Main
# -------------------------
def main() -> None:
    kb_dir = Path("data/knowledge_base").resolve()
    out_dir = Path("rag_index").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # recursive: supports subfolders later (e.g., knowledge_base/compressor/*.pdf)
    pdfs = sorted([p for p in kb_dir.rglob("*.pdf") if p.is_file()])

    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {kb_dir}. Add manuals there first.")

    print(f"[INFO] Knowledge base: {kb_dir}")
    print(f"[INFO] PDFs found: {len(pdfs)}")
    for p in pdfs:
        print(f"  - {p.relative_to(kb_dir)}")

    chunks: List[Chunk] = []

    for pdf in pdfs:
        doc_id = stable_doc_id(pdf)
        try:
            pages = read_pdf(pdf)
        except Exception as e:
            print(f"[WARN] Failed to read PDF: {pdf.name} | error={e}")
            continue

        for page_num, page_text in pages:
            parts = chunk_text(page_text, chunk_chars=900, overlap=150)
            for k, part in enumerate(parts):
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        source=pdf.name,  # keep original filename for citations
                        page=page_num,
                        chunk_id=f"{doc_id}_p{page_num}_c{k}",
                        text=part,
                    )
                )

    if not chunks:
        raise RuntimeError("No text chunks extracted. Your PDFs may be scanned images (no selectable text).")

    texts = [c.text for c in chunks]

    print(f"[INFO] Embedding {len(texts)} chunks...")
    vecs = embed_texts(texts)  # (n, d) float32 expected
    if not isinstance(vecs, np.ndarray):
        vecs = np.array(vecs, dtype=np.float32)

    #  ensure cosine-sim behavior for IndexFlatIP
    vecs = l2_normalize(vecs)

    n, d = vecs.shape
    index = faiss.IndexFlatIP(d)
    index.add(vecs)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    DocStore(chunks).save(out_dir / "chunks.json")

    print(f"\n Built RAG index: {out_dir}")
    print(f"- PDFs indexed: {len(pdfs)}")
    print(f"- Chunks: {n}")
    print(f"- Dim: {d}")
    print(f"- Files: {out_dir / 'faiss.index'} , {out_dir / 'chunks.json'}")


if __name__ == "__main__":
    main()
