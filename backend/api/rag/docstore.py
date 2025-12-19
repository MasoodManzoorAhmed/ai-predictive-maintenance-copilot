from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Chunk:
    doc_id: str
    source: str
    page: int
    chunk_id: str
    text: str


class DocStore:
    """
    Minimal docstore:
    - chunks.json contains list[Chunk]
    - ids are aligned with FAISS vectors order
    """

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks

    @staticmethod
    def load(path: Path) -> "DocStore":
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk(**x) for x in data]
        return DocStore(chunks)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data: List[Dict[str, Any]] = [asdict(c) for c in self.chunks]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
