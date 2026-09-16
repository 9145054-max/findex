import pickle
import json
from typing import Any

def save_pickle(obj: Any, filepath: str):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str) -> Any:
    with open(filepath, "rb") as f:
        return pickle.load(f)

def save_json(index_obj, filepath: str):
    data = {
        "doc_metadata": {k: {"doc_id": v.doc_id, "filename": v.filename, "length": v.length} for k, v in index_obj.doc_metadata.items()},
        "postings": {
            k: [{"doc_id": p.doc_id, "tf": p.tf, "positions": p.positions} for p in v]
            for k, v in index_obj.postings.items()
        }
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

