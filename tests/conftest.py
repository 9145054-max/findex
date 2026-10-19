import pytest
from pathlib import Path
from findex.index import InvertedIndex

@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.txt").write_text("python search engine ranking", encoding="utf-8")
    (corpus_dir / "doc2.txt").write_text("python async await search", encoding="utf-8")
    (corpus_dir / "doc3.txt").write_text("java spring framework", encoding="utf-8")
    return corpus_dir

@pytest.fixture
def built_index(sample_corpus: Path) -> InvertedIndex:
    idx = InvertedIndex(with_positions=True)
    idx.build(str(sample_corpus))
    return idx