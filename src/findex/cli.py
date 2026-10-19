import asyncio
import json
import time
from pathlib import Path
from enum import Enum
from typing import List
import typer
from rich.console import Console

from findex.crawler import crawl
from findex.index import InvertedIndex, build_partial, merge_indices

app = typer.Typer(name="Findex", help="Findex: Пошуковий індекс та система ранжування")
console = Console()


class ExecutorType(str, Enum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


@app.command()
def crawl_cmd(
    start_url: str = typer.Argument(..., help="Початкова URL адреса"),
    out: Path = typer.Option(Path("pages.jsonl"), help="Шлях для збереження JSONL"),
    max_pages: int = typer.Option(200, "--max-pages", "-m", help="Максимальна кількість сторінок"),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Паралельність запитів")
):
    """Асинхронний краулер сторінок з відображенням статистики в реальному часі."""
    start_time = time.time()
    fetched_count = 0

    async def run():
        nonlocal fetched_count
        with open(out, "w", encoding="utf-8") as f:
            async for page in crawl(start_url, max_pages=max_pages, concurrency=concurrency):
                f.write(json.dumps(page, ensure_ascii=False) + "\n")
                fetched_count += 1

    typer.echo(f"Запуск краулера для {start_url} (max_pages={max_pages}, concurrency={concurrency})...")
    asyncio.run(run())
    
    elapsed = time.time() - start_time
    typer.echo(f"Збірку завершено! Зібрано сторінок: {fetched_count} за {elapsed:.2f}s ({fetched_count/max(elapsed, 0.001):.2f} pages/s)")


@app.command()
def index(
    data_dir: Path = typer.Argument(..., help="Папка з документами"),
    out: Path = typer.Option(Path("index.bin"), help="Шлях для збереження індексу"),
    workers: int = typer.Option(1, "--workers", "-w", help="Кількість воркерів"),
    executor: ExecutorType = typer.Option(
        ExecutorType.serial, "--executor", "-e", help="Тип виконавця (serial, threads, processes)"
    ),
):
    """Побудова індексу із підтримкою паралелізму."""
    all_files = sorted([p for p in data_dir.rglob("*") if p.is_file()])
    if not all_files:
        typer.echo("Файли не знайдені.")
        raise typer.Exit(code=1)

    chunk_size = (len(all_files) + workers - 1) // workers
    tasks = []

    current_doc_id = 0
    for i in range(0, len(all_files), chunk_size):
        chunk_paths = all_files[i : i + chunk_size]
        tasks.append((chunk_paths, current_doc_id))
        current_doc_id += len(chunk_paths)

    partial_indices: List[InvertedIndex] = []

    if executor == ExecutorType.serial or workers == 1:
        for task in tasks:
            partial_indices.append(build_partial(task))
    elif executor == ExecutorType.threads:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            partial_indices = list(pool.map(build_partial, tasks))
    elif executor == ExecutorType.processes:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            partial_indices = list(pool.map(build_partial, tasks))

    t0 = time.perf_counter()
    final_index = merge_indices(partial_indices)
    t_merge = time.perf_counter() - t0

    typer.echo(
        f"Успішно побудовано індекс для {final_index.num_docs} документів. Час merge: {t_merge:.4f}s"
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="IP адреса для сервера"),
    port: int = typer.Option(8000, "--port", "-p", help="Порт сервера"),
    workers: int = typer.Option(1, "--workers", "-w", help="Кількість процесів uvicorn"),
    index_path: Path = typer.Option(Path("index.bin"), "--index-path", help="Шлях до індексу"),
):
    """Запуск FastAPI сервісу через Uvicorn."""
    import os
    import uvicorn

    os.environ["INDEX_PATH"] = str(index_path)
    typer.echo(f"Запуск Findex Web API на http://{host}:{port} (workers={workers})...")
    uvicorn.run("findex.web.app:app", host=host, port=port, workers=workers, reload=False)


if __name__ == "__main__":
    app()