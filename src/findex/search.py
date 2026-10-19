import sys
import time
import heapq
import functools
import tracemalloc
import re
from typing import List, Set, Tuple

from findex.index import open_index, InvertedIndex
from findex.models import SearchResult, Term, Phrase, And, Or, Not
from findex.scoring import Scorer, TfIdfScorer, BM25Scorer
from findex.parser import parse_query

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start = time.perf_counter()
        res = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[@timed] {func.__name__}: {elapsed:.6f} сек | Пам'ять: {peak / 1024:.2f} KB")
        return res
    return wrapper

@functools.lru_cache(maxsize=128)
def cached_search_doc_ids(query_str: str, index_path: str) -> Tuple[int, ...]:
    print(f"[LOG: Cache MISS] Виконується новий запит: '{query_str}'")
    with open_index(index_path) as index:
        ast = parse_query(query_str)
        doc_ids = evaluate_ast(ast, index)
        return tuple(sorted(doc_ids))

def evaluate_ast(node, index: InvertedIndex) -> Set[int]:
    all_docs = set(index.doc_metadata.keys())
    
    if isinstance(node, Term):
        return {p.doc_id for p in index[node.term]}
    elif isinstance(node, Phrase):
        if not node.terms:
            return set()
        first_postings = {p.doc_id: set(p.positions or []) for p in index[node.terms[0]]}
        matching_docs = set(first_postings.keys())

        for term_idx in range(1, len(node.terms)):
            term = node.terms[term_idx]
            current_postings = {p.doc_id: set(p.positions or []) for p in index[term]}
            matching_docs &= set(current_postings.keys())
            
            valid_docs = set()
            for doc_id in matching_docs:
                prev_pos = first_postings[doc_id]
                curr_pos = current_postings[doc_id]
                if any((pos + term_idx) in curr_pos for pos in prev_pos):
                    valid_docs.add(doc_id)
            matching_docs = valid_docs
        return matching_docs
    elif isinstance(node, And):
        return evaluate_ast(node.left, index) & evaluate_ast(node.right, index)
    elif isinstance(node, Or):
        return evaluate_ast(node.left, index) | evaluate_ast(node.right, index)
    elif isinstance(node, Not):
        return all_docs - evaluate_ast(node.operand, index)
    return set()

def make_snippet(doc_text: str, query_terms: List[str], window: int = 80) -> str:
    text_lower = doc_text.lower()
    best_pos = -1
    for term in query_terms:
        pos = text_lower.find(term)
        if pos != -1:
            best_pos = pos
            break

    if best_pos == -1:
        snippet = doc_text[:window * 2]
    else:
        start = max(0, best_pos - window)
        end = min(len(doc_text), best_pos + window)
        snippet = ("..." if start > 0 else "") + doc_text[start:end] + ("..." if end < len(doc_text) else "")

    for term in query_terms:
        if term:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            snippet = pattern.sub(lambda m: f"**{m.group(0)}**", snippet)
    return snippet

@timed
def search(query_str: str, index: InvertedIndex, scorer: Scorer, top_k: int = 5) -> List[SearchResult]:
    ast = parse_query(query_str)
    candidate_doc_ids = evaluate_ast(ast, index)
    query_terms = [t.lower() for t in re.findall(r'\w+', query_str) if t.upper() not in ("AND", "OR", "NOT")]
    
    scored_results = []
    for doc_id in candidate_doc_ids:
        doc_meta = index.doc_metadata[doc_id]
        total_score = 0.0

        for term in query_terms:
            postings = index[term]
            p = next((post for post in postings if post.doc_id == doc_id), None)
            if p:
                total_score += scorer.score_term(p.tf, index.df(term), doc_meta.length, index)

        try:
            with open(doc_meta.filename, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            snippet = make_snippet(content, query_terms)
        except Exception:
            snippet = ""

        scored_results.append(SearchResult(score=total_score, doc_id=doc_id, filename=doc_meta.filename, snippet=snippet))

    return heapq.nlargest(top_k, scored_results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ranked search with AST & BM25/TF-IDF")
    parser.add_argument("index_file", help="Index file path")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    with open_index(args.index_file) as index:
        scorer = BM25Scorer() if args.scorer == "bm25" else TfIdfScorer()
        results = search(args.query, index, scorer, top_k=args.top_k)

        print(f"\nРезультати пошуку ({args.scorer.upper()}):")
        for res in results:
            print(f"[{res.score:.4f}] Doc ID: {res.doc_id} ({res.filename})")
            print(f"  Сніпет: {res.snippet}\n")