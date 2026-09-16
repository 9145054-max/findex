import pytest
from findex.corpus import tokenize

def test_basic_tokenization():
    assert list(tokenize("Hello, World!")) == ["hello", "world"]

def test_apostrophes_and_hyphens():
    assert list(tokenize("Don't stop high-tech")) == ["don't", "stop", "high-tech"]

def test_unicode_normalization():
    assert list(tokenize("cafe CAFE")) == ["cafe", "cafe"]
