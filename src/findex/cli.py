import os
import time
from enum import Enum
from pathlib import Path
from typing import List
import typer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from findex.index import InvertedIndex, build_partial, merge_indices

app = typer.Typer(name="Findex", help="Findex: Пошуковий індекс та система ранжування")


class ExecutorType(str, Enum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


@app.command()
def index(
    data_dir: Path = typer.Argument(..., help="Папка з документами"),
    out: Path = typer.Option(Path("index.bin"), help="Шлях для збереження індексу"),
    workers: int = typer.Option(1, "--workers", "-w", help="Кількість воркерів"),
    executor: ExecutorType = typer.Option(
        ExecutorType.serial, "--executor", "-e", help="Тип виконавця (serial, threads, processes)"
    ),
):
    """Findex index builder command."""
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
        with ThreadPoolExecutor(max_workers=workers) as pool:
            partial_indices = list(pool.map(build_partial, tasks))
    elif executor == ExecutorType.processes:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            partial_indices = list(pool.map(build_partial, tasks))

    t0 = time.perf_counter()
    final_index = merge_indices(partial_indices)
    t_merge = time.perf_counter() - t0

    typer.echo(
        f"Успішно побудовано індекс для {final_index.num_docs} документів. Час merge: {t_merge:.4f}s"
    )


if __name__ == "__main__":
    app()