import pickle
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import numpy as np

try:
    from findex.tokenizer import tokenize
except ImportError:
    try:
        from findex import tokenize_text as tokenize
    except ImportError:
        def tokenize(text: str) -> List[str]:
            import re
            return [w.lower() for w in re.findall(r"\w+", text)]


class InvertedIndex:
    def __init__(self, with_positions: bool = False):
        self.with_positions = with_positions
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_lengths: Dict[int, int] = {}
        self.doc_texts: Dict[int, str] = {}
        self.doc_paths: Dict[int, Path] = {}
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0

    @property
    def index(self):
        return self.postings

    def __contains__(self, item: str) -> bool:
        return item in self.postings

    def __len__(self) -> int:
        return len(self.postings)

    def add_document(self, doc_id: int, text: str, doc_path: Optional[Path] = None):
        tokens = tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_texts[doc_id] = text
        if doc_path:
            self.doc_paths[doc_id] = doc_path

        counts = defaultdict(int)
        for token in tokens:
            counts[token] += 1

        for token, count in counts.items():
            self.postings[token].append((doc_id, count))

        self.num_docs = len(self.doc_lengths)
        if self.num_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.num_docs

    def build(self, documents: Union[Dict[int, str], str, Path]):
        if isinstance(documents, (str, Path)):
            p = Path(documents)
            if p.is_dir():
                all_files = sorted([f for f in p.rglob("*") if f.is_file()])
                for idx, file_path in enumerate(all_files):
                    try:
                        text = file_path.read_text(encoding="utf-8", errors="ignore")
                        self.add_document(idx, text, doc_path=file_path)
                    except Exception:
                        pass
        elif isinstance(documents, dict):
            for doc_id, text in documents.items():
                self.add_document(doc_id, text)

    def search(self, query: str, top_k: int = 10, k1: float = 1.5, b: float = 0.75) -> List[Tuple[int, float]]:
        query_tokens = tokenize(query)
        if not query_tokens or self.num_docs == 0:
            return []

        scores = np.zeros(self.num_docs, dtype=np.float64)
        doc_lens = np.array([self.doc_lengths.get(i, 1) for i in range(self.num_docs)], dtype=np.float32)

        for token in query_tokens:
            if token not in self.postings:
                continue

            posting_list = self.postings[token]
            df = len(posting_list)
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

            doc_ids = np.array([p[0] for p in posting_list], dtype=np.int32)
            tfs = np.array([p[1] for p in posting_list], dtype=np.float32)

            valid_mask = doc_ids < self.num_docs
            doc_ids = doc_ids[valid_mask]
            tfs = tfs[valid_mask]

            lens = doc_lens[doc_ids]
            denom = tfs + k1 * (1.0 - b + b * (lens / (self.avg_doc_length or 1.0)))
            bm25_vals = idf * (tfs * (k1 + 1.0)) / denom

            np.add.at(scores, doc_ids, bm25_vals)

        nonzero_indices = np.flatnonzero(scores)
        if len(nonzero_indices) == 0:
            return []

        if len(nonzero_indices) > top_k:
            partition_idx = np.argpartition(scores[nonzero_indices], -top_k)[-top_k:]
            top_indices = nonzero_indices[partition_idx]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        else:
            top_indices = nonzero_indices[np.argsort(-scores[nonzero_indices])]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def get_document_text(self, doc_id: int) -> str:
        return self.doc_texts.get(doc_id, "")

    def save(self, filepath: Path):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "InvertedIndex":
        with open(filepath, "rb") as f:
            return pickle.load(f)


def build_partial(args) -> InvertedIndex:
    paths, start_doc_id = args
    index = InvertedIndex()
    for idx, path in enumerate(paths):
        doc_id = start_doc_id + idx
        try:
            p = Path(path) if isinstance(path, str) else path
            text = p.read_text(encoding="utf-8", errors="ignore")
            index.add_document(doc_id, text, doc_path=p)
        except Exception:
            pass
    return index


def merge_indices(indices: List[InvertedIndex]) -> InvertedIndex:
    final = InvertedIndex()
    for idx in indices:
        for token, plist in idx.postings.items():
            final.postings[token].extend(plist)
        final.doc_lengths.update(idx.doc_lengths)
        final.doc_texts.update(idx.doc_texts)
        final.doc_paths.update(idx.doc_paths)

    final.num_docs = len(final.doc_lengths)
    if final.num_docs > 0:
        final.avg_doc_length = sum(final.doc_lengths.values()) / final.num_docs
    return final
