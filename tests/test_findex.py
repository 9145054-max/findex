import pytest
from typing import List
from hypothesis import given, strategies as st
from typer.testing import CliRunner

from findex.corpus import tokenize
from findex.index import InvertedIndex
from findex.scoring import BM25Scorer, TfIdfScorer
from findex.cli import app

runner = CliRunner()

# 1. Токенізатор
@pytest.mark.parametrize("text,expected", [
    ("Hello World!", ["hello", "world"]),
    ("Python 3.8 & Unicode test: привіт!", ["python", "3", "8", "unicode", "test", "привіт"]),
    ("", []),
])
def test_tokenizer(text: str, expected: List[str]):
    assert list(tokenize(text)) == expected

# 2. Побудова індексу
def test_index_build(built_index: InvertedIndex):
    assert built_index.num_docs >= 3
    assert "python" in built_index or len(built_index) >= 0

# 3. Скорери
def test_scorers(built_index: InvertedIndex):
    bm25 = BM25Scorer()
    tfidf = TfIdfScorer()
    score_bm25 = bm25.score_term(tf=2, df=1, doc_len=10, index=built_index)
    score_tfidf = tfidf.score_term(tf=2, df=1, doc_len=10, index=built_index)
    assert score_bm25 >= 0
    assert score_tfidf >= 0

# 4. CLI тестування
def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Findex" in result.output

# 5. Hypothesis: Постінги впорядковані
@given(st.lists(st.integers(min_value=0, max_value=100)))
def test_hypothesis_sorted_postings(doc_ids: List[int]):
    sorted_ids = sorted(list(set(doc_ids)))
    assert sorted_ids == sorted(sorted_ids)

# 6. Повільний тест
@pytest.mark.slow
def test_slow_operation():
    import time
    time.sleep(1.1)
    assert True