from pathlib import Path
from typing import List, Tuple, Dict, Iterable
from collections import defaultdict
from findex.corpus import tokenize


class InvertedIndex:
    def __init__(self, with_positions: bool = False):
        self.with_positions = with_positions
        self.index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_lengths: Dict[int, int] = {}
        self.doc_paths: Dict[int, str] = {}
        self.num_docs: int = 0

    @property
    def avg_doc_length(self) -> float:
        """Середня довжина документа у корпусі."""
        if self.num_docs == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / self.num_docs

    def __contains__(self, term: str) -> bool:
        """Підтримка перевірки 'term in index'."""
        return term in self.index

    def __iter__(self):
        """Ітерація по термінах індексу."""
        return iter(self.index)

    def add_document(self, doc_id: int, path: str, terms: Iterable[str]):
        term_freqs: Dict[str, int] = defaultdict(int)
        length = 0
        for term in terms:
            term_freqs[term] += 1
            length += 1

        self.doc_lengths[doc_id] = length
        self.doc_paths[doc_id] = str(path)
        self.num_docs += 1

        for term, freq in term_freqs.items():
            self.index[term].append((doc_id, freq))

    def build(self, data_dir: str):
        """Метод для сумісності зі старими тестами."""
        path = Path(data_dir)
        files = sorted([p for p in path.rglob("*") if p.is_file()])
        for doc_id, file_path in enumerate(files):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                tokens = tokenize(text)
                self.add_document(doc_id, str(file_path), tokens)
            except Exception:
                continue

    def merge(self, other: "InvertedIndex"):
        """Об'єднання іншої частини індексу в поточний."""
        for doc_id, length in other.doc_lengths.items():
            self.doc_lengths[doc_id] = length
            self.doc_paths[doc_id] = other.doc_paths[doc_id]

        self.num_docs += other.num_docs

        for term, postings in other.index.items():
            self.index[term].extend(postings)
            self.index[term].sort(key=lambda x: x[0])


def build_partial(args: Tuple[List[Path], int]) -> InvertedIndex:
    """
    Функція обробки шматка файлів.
    Приймає (список_шляхів, start_doc_id).
    """
    paths, start_doc_id = args
    partial_index = InvertedIndex()

    for idx, path in enumerate(paths):
        doc_id = start_doc_id + idx
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            tokens = tokenize(text)
            partial_index.add_document(doc_id, str(path), tokens)
        except Exception:
            continue

    return partial_index


def merge_indices(indices: List[InvertedIndex]) -> InvertedIndex:
    """Послідовне об'єднання всіх часткових індексів."""
    main_index = InvertedIndex()
    for idx in indices:
        main_index.merge(idx)
    return main_index