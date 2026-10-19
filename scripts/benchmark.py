import os
import time
import psutil
import statistics
import matplotlib.pyplot as plt
from pathlib import Path
from findex.index import build_partial, merge_indices
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Використовуємо тестовий корпус з папки data або будь-якої іншої папки з документами
DATA_DIR = Path("data")


def run_experiment(workers: int, executor_type: str):
    all_files = sorted([p for p in DATA_DIR.rglob("*") if p.is_file()])
    if not all_files:
        print("Помилка: папка 'data' порожня або не існує!")
        return 0, 0, 0, 0

    chunk_size = (len(all_files) + workers - 1) // workers
    tasks = []

    current_doc_id = 0
    for i in range(0, len(all_files), chunk_size):
        chunk_paths = all_files[i : i + chunk_size]
        tasks.append((chunk_paths, current_doc_id))
        current_doc_id += len(chunk_paths)

    process = psutil.Process(os.getpid())

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()

    if executor_type == "serial" or workers == 1:
        partials = [build_partial(t) for t in tasks]
    elif executor_type == "threads":
        with ThreadPoolExecutor(max_workers=workers) as ex:
            partials = list(ex.map(build_partial, tasks))
    elif executor_type == "processes":
        with ProcessPoolExecutor(max_workers=workers) as ex:
            partials = list(ex.map(build_partial, tasks))

    t0_merge = time.perf_counter()
    final_index = merge_indices(partials)
    merge_time = time.perf_counter() - t0_merge

    wall_time = time.perf_counter() - t0_wall
    cpu_time = time.process_time() - t0_cpu
    mem_rss = process.memory_info().rss / (1024 * 1024)  # у МБ

    return wall_time, cpu_time, mem_rss, merge_time


def main():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Створюємо кілька тестових файлів, якщо папка була порожня
        for i in range(100):
            (DATA_DIR / f"doc_{i}.txt").write_text(
                f"Sample text document number {i} with some concurrency and GIL tests.",
                encoding="utf-8",
            )

    cpu_cnt = os.cpu_count() or 4
    workers_list = sorted(list(set([1, 2, 4, 8, cpu_cnt])))

    print("=== Запуск бенчмарку Лабораторної роботи 5 ===")
    print(f"Робочі воркери: {workers_list}\n")

    results = {}
    for mode in ["threads", "processes"]:
        results[mode] = []
        print(f"--- Тестування {mode} ---")
        for w in workers_list:
            runs = [run_experiment(w, mode) for _ in range(3)]
            med_wall = statistics.median([r[0] for r in runs])
            med_cpu = statistics.median([r[1] for r in runs])
            med_rss = statistics.median([r[2] for r in runs])
            med_merge = statistics.median([r[3] for r in runs])
            results[mode].append((w, med_wall, med_cpu, med_rss, med_merge))
            print(
                f"Workers: {w} | Wall Time: {med_wall:.4f}s | CPU Time: {med_cpu:.4f}s | RSS: {med_rss:.2f}MB | Merge: {med_merge:.4f}s"
            )

    # Генерація графіку прискорення
    base_wall_proc = results["processes"][0][1]
    speedups_proc = [
        base_wall_proc / r[1] if r[1] > 0 else 1.0 for r in results["processes"]
    ]

    base_wall_thr = results["threads"][0][1]
    speedups_thr = [
        base_wall_thr / r[1] if r[1] > 0 else 1.0 for r in results["threads"]
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(
        workers_list, speedups_proc, "o-", label="Processes (ProcessPoolExecutor)"
    )
    plt.plot(
        workers_list, speedups_thr, "s--", label="Threads (ThreadPoolExecutor)"
    )
    plt.plot(workers_list, workers_list, "k:", label="Ideal Linear Speedup")
    plt.xlabel("Кількість воркерів (Workers)")
    plt.ylabel("Прискорення (Speedup)")
    plt.title("Лабораторна 5: Залежність прискорення від кількості воркерів")
    plt.legend()
    plt.grid(True)
    plt.savefig("speedup.png")
    print("\nГрафік успішно збережено у файл speedup.png!")


if __name__ == "__main__":
    main()