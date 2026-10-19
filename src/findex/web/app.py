import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Query, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from findex.web.config import settings
from findex.web.schemas import (
    SearchResponse,
    SearchResultItem,
    DocumentResponse,
    StatsResponse,
    HealthResponse,
)
from findex.index import InvertedIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("findex.web")

# Глобальне сховище індексу в пам'яті
app_state = {"index": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lifespan: завантажуємо індекс один раз при запуску
    if settings.INDEX_PATH.exists():
        logger.info(f"Завантаження індексу з {settings.INDEX_PATH}...")
        try:
            app_state["index"] = InvertedIndex.load(settings.INDEX_PATH)
            logger.info("Індекс успішно завантажено!")
        except Exception as e:
            logger.error(f"Помилка завантаження індексу: {e}")
            app_state["index"] = None
    else:
        logger.warning(f"Файл індексу {settings.INDEX_PATH} не знайдено.")
    yield
    app_state["index"] = None

app = FastAPI(title="Findex Web Search API", lifespan=lifespan)

# Шаблони Jinja2
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

def get_index():
    index = app_state.get("index")
    if index is None:
        raise HTTPException(status_code=503, detail="Index is not loaded")
    return index

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    req_id = str(uuid.uuid4())
    logger.error(f"Request {req_id} failed with exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "request_id": req_id},
    )

@app.get("/health", response_model=HealthResponse)
def health_check():
    if app_state.get("index") is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Index not loaded")
    return HealthResponse(status="ok")

@app.get("/stats", response_model=StatsResponse)
def stats(index: InvertedIndex = Depends(get_index)):
    return StatsResponse(
        total_documents=getattr(index, "num_docs", 0),
        index_path=str(settings.INDEX_PATH),
    )

@app.get("/search", response_model=SearchResponse)
def search_api(
    q: str = Query(..., min_length=1, description="Пошуковий запит"),
    k: int = Query(10, ge=1, le=100),
    page: int = Query(1, ge=1),
    scorer: str = Query("bm25", pattern="^(bm25|tfidf)$"),
    index: InvertedIndex = Depends(get_index),
):
    # Пошуковий виклик (def використовує threadpool uvicorn для блокуючих дій)
    raw_results = getattr(index, "search", lambda query, top_k: [])(q, top_k=k * page)
    
    start_idx = (page - 1) * k
    end_idx = start_idx + k
    page_results = raw_results[start_idx:end_idx]

    items = []
    for doc_id, score in page_results:
        items.append(
            SearchResultItem(
                doc_id=doc_id,
                score=round(score, 4),
                title=f"Document #{doc_id}",
                snippet=f"Результат для запиту '{q}' у документі #{doc_id}..."
            )
        )

    return SearchResponse(
        query=q,
        total_hits=len(raw_results),
        page=page,
        page_size=k,
        results=items,
    )

@app.get("/docs/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: int, index: InvertedIndex = Depends(get_index)):
    if hasattr(index, "get_document_text"):
        text = index.get_document_text(doc_id)
        if text is None:
            raise HTTPException(status_code=404, detail=f"Document #{doc_id} not found")
        return DocumentResponse(doc_id=doc_id, content=text)
    
    if doc_id < 0 or doc_id >= getattr(index, "num_docs", 0):
        raise HTTPException(status_code=404, detail=f"Document #{doc_id} not found")
    
    return DocumentResponse(doc_id=doc_id, content=f"Текст документа #{doc_id}")

@app.get("/", response_class=HTMLResponse)
def search_ui(
    request: Request,
    q: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
):
    results = None
    if q and app_state.get("index") is not None:
        index = app_state["index"]
        raw_results = getattr(index, "search", lambda query, top_k: [])(q, top_k=10 * page)
        results = [
            {"doc_id": doc_id, "score": round(score, 4), "title": f"Документ #{doc_id}"}
            for doc_id, score in raw_results[(page - 1) * 10 : page * 10]
        ]

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "query": q or "", "results": results, "page": page},
    )