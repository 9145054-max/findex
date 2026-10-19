import pytest
from pathlib import Path
from findex.index import InvertedIndex, build_partial, merge_indices


def test_build_partial_and_merge_equivalence(tmp_path: Path):
    # Створюємо декілька тестових файлів
    doc1 = tmp_path / "doc1.txt"
    doc2 = tmp_path / "doc2.txt"
    doc1.write_text("python asyncio testing", encoding="utf-8")
    doc2.write_text("python GIL concurrency multiprocessing", encoding="utf-8")

    files = [doc1, doc2]

    # 1. Послідовне створення одного індексу
    full_task = (files, 0)
    serial_index = build_partial(full_task)

    # 2. Розбиття на частини (по 1 файлу на воркер)
    task1 = ([doc1], 0)
    task2 = ([doc2], 1)

    partial1 = build_partial(task1)
    partial2 = build_partial(task2)

    merged_index = merge_indices([partial1, partial2])

    # Перевіряємо еквівалентність
    assert merged_index.num_docs == serial_index.num_docs == 2
    assert merged_index.doc_lengths == serial_index.doc_lengths
    assert merged_index.doc_paths == serial_index.doc_paths
    
    # Перевіряємо, що всі словникові ключі та постінги збігаються
    assert set(merged_index.index.keys()) == set(serial_index.index.keys())
    for term in merged_index.index:
        assert merged_index.index[term] == serial_index.index[term]