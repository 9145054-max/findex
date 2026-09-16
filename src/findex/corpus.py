import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Generator, List, Tuple, Union


def iter_documents(data_dir: Union[str, Path]) -> Generator[Tuple[str, str], None, None]:
    data_path = Path(data_dir)
    for file_path in data_path.rglob("*.txt"):
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    yield file_path.name, f.read()
            except Exception as e:
                print(f"Помилка читання {file_path}: {e}")


def tokenize(text: str) -> Generator[str, None, None]:
    text = unicodedata.normalize("NFKC", text)
    pattern = re.compile(r"\b\w+(?:[''-]\w+)*\b")

    for match in pattern.finditer(text.lower()):
        yield match.group(0)


def process_corpus_lazy(data_dir: Union[str, Path]) -> Tuple[int, int, int, List[Tuple[str, int]]]:
    doc_count = 0
    token_count = 0
    vocab = Counter()

    for doc_name, content in iter_documents(data_dir):
        doc_count += 1
        for token in tokenize(content):
            token_count += 1
            vocab[token] += 1

    return doc_count, token_count, len(vocab), vocab.most_common(50)


def process_corpus_eager(data_dir: Union[str, Path]) -> Tuple[int, int, int, List[Tuple[str, int]]]:
    data_path = Path(data_dir)
    all_documents = []

    for file_path in data_path.rglob("*.txt"):
        if file_path.is_file():
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                all_documents.append(f.read())

    all_tokens = []
    for doc in all_documents:
        all_tokens.extend(list(tokenize(doc)))

    vocab = Counter(all_tokens)

    return len(all_documents), len(all_tokens), len(vocab), vocab.most_common(50)