import sys
import time
import tracemalloc
from collections import defaultdict
from typing import Dict, List, Tuple
from findex.corpus import process_corpus_lazy, tokenize
from findex.models import Posting, DocMeta

class InvertedIndex:
    def __init__(self, with_positions: bool = False):
        self.with_positions = with_positions
        self.postings: Dict[str, List[Posting]] = defaultdict(list)
        self.doc_metadata: Dict[int, DocMeta] = {}

    def build(self, corpus_path: str):
        iterator = iter(process_corpus_lazy(corpus_path))
        
        while True:
            try:
                first_val = next(iterator)
            except StopIteration:
                break

            # Якщо генератор повертає кортеж/список (doc_id, filepath, content)
            if isinstance(first_val, (tuple, list)):
                doc_id = first_val[0]
                filepath = first_val[1]
                content = first_val[2] if len(first_val) > 2 else ""
            # Якщо генератор повертає окремі елементи через yield
            else:
                doc_id = first_val
                filepath = next(iterator)
                content = next(iterator)

            # Перевіряємо, чи content - це вже список токенів, чи текст
            if isinstance(content, (list, tuple)):
                tokens = content
            else:
                tokens = list(tokenize(str(content)))

            term_freqs: Dict[str, int] = defaultdict(int)
            term_positions: Dict[str, List[int]] = defaultdict(list)
            doc_len = 0

            for pos, token in enumerate(tokens):
                doc_len += 1
                term_freqs[token] += 1
                if self.with_positions:
                    term_positions[token].append(pos)

            filename = str(filepath)

            self.doc_metadata[doc_id] = DocMeta(
                doc_id=doc_id,
                filename=filename,
                length=doc_len
            )

            for term, tf in term_freqs.items():
                pos_tuple = tuple(term_positions[term]) if self.with_positions else None
                self.postings[term].append(Posting(doc_id=doc_id, tf=tf, positions=pos_tuple))

if __name__ == "__main__":
    import argparse
    from findex.storage import save_pickle

    parser = argparse.ArgumentParser(description="Build and save inverted index")
    parser.add_argument("data_dir", help="Path to corpus directory")
    parser.add_argument("--out", default="index.bin", help="Output index file")
    parser.add_argument("--positions", action="store_true", help="Store token positions")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    idx = InvertedIndex(with_positions=args.positions)
    idx.build(args.data_dir)

    save_pickle(idx, args.out)

    elapsed = time.perf_counter() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Індекс побудовано за {elapsed:.4f} сек")
    print(f"Пікова пам'ять: {peak_mem / (1024 * 1024):.2f} МБ")