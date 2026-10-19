import math
from typing import Protocol, runtime_checkable
from findex.index import InvertedIndex

@runtime_checkable
class Scorer(Protocol):
    def score_term(self, tf: int, df: int, doc_len: int, index: InvertedIndex) -> float:
        ...

class TfIdfScorer:
    def score_term(self, tf: int, df: int, doc_len: int, index: InvertedIndex) -> float:
        if tf <= 0 or df <= 0 or index.num_docs == 0:
            return 0.0
        tf_val = 1 + math.log10(tf)
        idf_val = math.log10(index.num_docs / df)
        return tf_val * idf_val

class BM25Scorer:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score_term(self, tf: int, df: int, doc_len: int, index: InvertedIndex) -> float:
        if tf <= 0 or index.num_docs == 0:
            return 0.0
        
        N = index.num_docs
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        
        avgdl = index.avg_doc_length
        len_norm = 1.0 - self.b + self.b * (doc_len / avgdl) if avgdl > 0 else 1.0
        
        tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
        return idf * tf_component