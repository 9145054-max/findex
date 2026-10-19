import sys
import time
import tracemalloc
from collections.abc import Mapping
from functools import cached_property
from typing import Dict, List, Iterator, Optional, Any
from contextlib import contextmanager

from findex.corpus import process_corpus_lazy, tokenize
from findex.models import Posting, DocMeta
from findex.storage import load_pickle, save_pickle

class InvertedIndex(Mapping):
    def __init__(self, with_positions: bool = True):
        self.with_positions = with_positions
        self._postings: Dict[str, List[Posting]] = {}
        self.doc_metadata: Dict[int, DocMeta] = {}

    def __getitem__(self, term: str) -> List[Posting]:
        return self._postings.get(term, [])

    def __len__(self) -> int:
        return len(self._postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __contains__(self, term: object) -> bool:
        return term in self._postings

    def __repr__(self) -> str:
        return f"<InvertedIndex terms={len(self)}, docs={self.num_docs}>"

    @property
    def num_docs(self) -> int:
        return len(self.doc_metadata)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self.doc_metadata:
            return 0.0
        total_len = sum(doc.length for doc in self.doc_metadata.values())
        return total_len / len(self.doc_metadata)

    def df(self, term: str) -> int:
        return len(self._postings.get(term, []))

    def build(self, corpus_path: str):
        raw_postings = {}
        doc_counter = 0

        for item in process_corpus_lazy(corpus_path):
            if isinstance(item, tuple) and len(item) >= 2:
                filepath = item[0]
                content = item[1]
            else:
                filepath = str(item)
                content = ""

            doc_id = doc_counter
            doc_counter += 1

            tokens = list(tokenize(str(content))) if isinstance(content, str) else list(content)

            term_freqs = {}
            term_positions = {}
            doc_len = len(tokens)

            for pos, token in enumerate(tokens):
                term_freqs[token] = term_freqs.get(token, 0) + 1
                if self.with_positions:
                    if token not in term_positions:
                        term_positions[token] = []
                    term_positions[token].append(pos)

            self.doc_metadata[doc_id] = DocMeta(
                doc_id=doc_id,
                filename=str(filepath),
                length=doc_len
            )

            for term, tf in term_freqs.items():
                if term not in raw_postings:
                    raw_postings[term] = []
                pos_tuple = tuple(term_positions[term]) if self.with_positions else None
                raw_postings[term].append(Posting(doc_id=doc_id, tf=tf, positions=pos_tuple))

        self._postings = raw_postings

@contextmanager
def open_index(path: str):
    index_obj = load_pickle(path)
    try:
        yield index_obj
    finally:
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build index with positions")
    parser.add_argument("data_dir", help="Path to corpus")
    parser.add_argument("--out", default="index.bin", help="Output file")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    idx = InvertedIndex(with_positions=True)
    idx.build(args.data_dir)
    save_pickle(idx, args.out)

    elapsed = time.perf_counter() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Індекс побудовано за {elapsed:.4f} сек | Пам'ять: {peak_mem / (1024 * 1024):.2f} МБ")