from findex.index import InvertedIndex
import argparse
import time
import tracemalloc
from typing import List
from findex.corpus import tokenize
from findex.storage import load_pickle

def intersect_two_postings_merge(p1: List[int], p2: List[int]) -> List[int]:
    i, j = 0, 0
    result = []
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            result.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1
    return result

def intersect_two_postings_set(p1: List[int], p2: List[int]) -> List[int]:
    return sorted(list(set(p1) & set(p2)))

def search_query(index, query: str, engine: str = "merge") -> List[int]:
    tokens = list(tokenize(query))
    if not tokens:
        return []

    doc_lists = []
    for t in tokens:
        postings = index.postings.get(t, [])
        doc_lists.append([p.doc_id for p in postings])

    if not doc_lists or any(len(lst) == 0 for lst in doc_lists):
        return []

    doc_lists.sort(key=len)
    result = doc_lists[0]
    intersect_fn = intersect_two_postings_merge if engine == "merge" else intersect_two_postings_set

    for next_list in doc_lists[1:]:
        result = intersect_fn(result, next_list)
        if not result:
            break

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search in inverted index")
    parser.add_argument("index_file", help="Path to index file")
    parser.add_argument("query", help="Query string")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge", help="Intersect engine")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    idx = load_pickle(args.index_file)
    results = search_query(idx, args.query, engine=args.engine)

    elapsed = time.perf_counter() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Результати пошуку ({args.engine}): {results}")
    print(f"Знайдено документів: {len(results)}")
    print(f"Час виконання: {elapsed:.6f} сек")
    print(f"Пікова пам'ять: {peak_mem / (1024 * 1024):.2f} МБ")