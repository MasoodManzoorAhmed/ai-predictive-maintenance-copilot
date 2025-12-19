from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .docstore import Chunk, DocStore
from .embedder import embed_texts


@dataclass
class RetrievedChunk:
    score: float
    chunk: Chunk


class RAGRetriever:
    def __init__(self, index, docstore: DocStore):
        self.index = index
        self.docstore = docstore

    @staticmethod
    def load(index_dir: Path) -> "RAGRetriever":
        import faiss  # type: ignore

        index_path = index_dir / "faiss.index"
        chunks_path = index_dir / "chunks.json"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"RAG index missing. Expected: {index_path} and {chunks_path} "
                f"(build it with scripts/build_rag_index.py)"
            )

        index = faiss.read_index(str(index_path))
        docstore = DocStore.load(chunks_path)
        return RAGRetriever(index=index, docstore=docstore)

    def search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        # Embed query -> ensure float32 numpy array with shape (1, d)
        qvec = embed_texts([query])

        if isinstance(qvec, list):
            qvec = np.array(qvec, dtype=np.float32)
        else:
            qvec = np.asarray(qvec, dtype=np.float32)

        if qvec.ndim != 2 or qvec.shape[0] != 1:
            raise ValueError(f"embed_texts output must be 2D with shape (1, d). Got shape: {qvec.shape}")

        # FAISS returns (distances, indices)
        distances, idxs = self.index.search(qvec, top_k)

        out: List[RetrievedChunk] = []
        for dist, i in zip(distances[0].tolist(), idxs[0].tolist()):
            if i < 0 or i >= len(self.docstore.chunks):
                continue
            out.append(RetrievedChunk(score=float(dist), chunk=self.docstore.chunks[i]))
        return out
