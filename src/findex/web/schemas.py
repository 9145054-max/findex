from pydantic import BaseModel, Field
from typing import List, Optional

class SearchResultItem(BaseModel):
    doc_id: int
    score: float
    title: Optional[str] = ""
    snippet: Optional[str] = ""

class SearchResponse(BaseModel):
    query: str
    total_hits: int
    page: int
    page_size: int
    results: List[SearchResultItem]

class DocumentResponse(BaseModel):
    doc_id: int
    content: str

class StatsResponse(BaseModel):
    total_documents: int
    index_path: str

class HealthResponse(BaseModel):
    status: str