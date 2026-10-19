import sys
import json
import logging
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from findex.index import InvertedIndex, open_index
from findex.scoring import BM25Scorer, TfIdfScorer
from findex.parser import parse_query
from findex.storage import save_pickle

app = typer.Typer(help="Findex: Пошуковий індекс та система ранжування")
console = Console(stderr=True)
out_console = Console()

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr
    )

@app.callback()
def main(
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Рівень деталізації логів (-v або -vv)")
) -> None:
    setup_logging(verbose)

@app.command()
def index(
    data_dir: Path = typer.Argument(..., help="Шлях до каталогу з документами"),
    out: Path = typer.Option(Path("index.bin"), "--out", "-o", help="Шлях до файлу індексу")
) -> None:
    if not data_dir.exists():
        logging.error(f"Каталог {data_dir} не існує.")
        raise typer.Exit(code=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task(description="Побудова індексу...", total=None)
        idx = InvertedIndex(with_positions=True)
        idx.build(str(data_dir))
        save_pickle(idx, str(out))

    console.print(f"[bold green]Успішно![/bold green] Індекс збережено у {out}")

@app.command()
def search(
    index_file: Path = typer.Argument(..., help="Файл індексу"),
    query: str = typer.Argument(..., help="Пошуковий запит"),
    scorer_type: str = typer.Option("bm25", "--scorer", "-s", help="Алгоритм ранжування (bm25 або tfidf)"),
    as_json: bool = typer.Option(False, "--json", help="Вивід у форматі JSON")
) -> None:
    if not index_file.exists():
        console.print(f"[bold red]Помилка:[/bold red] Файл індексу '{index_file}' не знайдено.")
        raise typer.Exit(code=1)

    try:
        ast = parse_query(query)
    except Exception as e:
        console.print(f"[bold red]Помилка парсингу запиту:[/bold red] {e}")
        raise typer.Exit(code=1)

    with open_index(str(index_file)) as idx:
        scorer = BM25Scorer() if scorer_type.lower() == "bm25" else TfIdfScorer()
        matching_docs = ast.evaluate(idx)
        
        results = []
        for doc_id in matching_docs:
            meta = idx.doc_metadata.get(doc_id)
            filename = meta.filename if meta else f"Doc_{doc_id}"
            results.append({"doc_id": doc_id, "filename": filename, "score": 1.0})

        if as_json:
            out_console.print(json.dumps(results, ensure_ascii=False, indent=2))
            return

        table = Table(title=f"Результати пошуку для: '{query}'")
        table.add_column("Doc ID", justify="right", style="cyan")
        table.add_column("Файл", style="magenta")
        table.add_column("Score", justify="right", style="green")

        for r in results:
            table.add_row(str(r["doc_id"]), r["filename"], f"{r['score']:.4f}")

        out_console.print(table)

@app.command()
def stats(
    index_file: Path = typer.Argument(..., help="Файл індексу")
) -> None:
    if not index_file.exists():
        console.print(f"[bold red]Помилка:[/bold red] Файл '{index_file}' не знайдено.")
        raise typer.Exit(code=1)

    with open_index(str(index_file)) as idx:
        console.print(f"[bold]Кількість документів:[/bold] {idx.num_docs}")
        console.print(f"[bold]Кількість унікальних термів:[/bold] {len(idx)}")
        console.print(f"[bold]Середня довжина документа:[/bold] {idx.avg_doc_length:.2f}")

if __name__ == "__main__":
    app()