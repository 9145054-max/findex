import pytest
from fastapi.testclient import TestClient
from findex.web.app import app, app_state
from findex.index import InvertedIndex

class DummyIndex:
    num_docs = 5
    def search(self, query, top_k=10):
        return [(0, 1.5), (1, 0.8)]
    def get_document_text(self, doc_id):
        if doc_id == 0:
            return "Test Document Content"
        return None

def test_health_503_when_no_index():
    app_state["index"] = None
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 503

def test_health_200_when_index_loaded():
    app_state["index"] = DummyIndex()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_search_success():
    app_state["index"] = DummyIndex()
    client = TestClient(app)
    response = client.get("/search?q=test")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test"
    assert len(data["results"]) == 2

def test_search_422_validation_error():
    app_state["index"] = DummyIndex()
    client = TestClient(app)
    response = client.get("/search?q=")  # Порожній запит
    assert response.status_code == 422

def test_doc_not_found_404():
    app_state["index"] = DummyIndex()
    client = TestClient(app)
    response = client.get("/docs/999")
    assert response.status_code == 404