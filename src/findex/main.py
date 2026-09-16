import time
import tracemalloc
from pathlib import Path
from findex.corpus import process_corpus_eager, process_corpus_lazy


def run_benchmark(mode: str, data_dir: Path):
    tracemalloc.start()
    start_time = time.perf_counter()

    if mode == "lazy":
        docs, tokens, unique, top50 = process_corpus_lazy(data_dir)
    else:
        docs, tokens, unique, top50 = process_corpus_eager(data_dir)

    elapsed_time = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)

    print(f"=== Режим: {mode.upper()} ===")
    print(f"Документів: {docs}")
    print(f"Всього токенів: {tokens}")
    print(f"Унікальних токенів: {unique}")
    print(f"Час: {elapsed_time:.4f} сек")
    print(f"Пікова пам'ять: {peak_mb:.2f} МБ")
    print("-" * 30)

    return elapsed_time, peak_mb


def main():
    data_dir = Path("data")
    if not data_dir.exists():
        print("Папку data не знайдено!")
        return

    print("Запуск вимірювань...\n")
    run_benchmark("lazy", data_dir)
    run_benchmark("eager", data_dir)


if __name__ == "__main__":
    main()