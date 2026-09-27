import numpy as np

from app.services.vectorstore import VectorStore
from app.services.chunking import Chunk


def buat_chunk_dummy(chunk_id, teks, is_active=True, doc_version="2.0"):
    return Chunk(
        chunk_id=chunk_id,
        text=teks,
        doc_id="DOC1",
        doc_title="Dokumen Test",
        doc_version=doc_version,
        is_active=is_active,
        section_title="Bagian Test",
    )


def test_build_dan_search_vectorstore():
    chunks = [
        buat_chunk_dummy("c1", "teks pertama"),
        buat_chunk_dummy("c2", "teks kedua"),
        buat_chunk_dummy("c3", "arsip lama", is_active=False, doc_version="1.4"),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    store = VectorStore(index_dir="/tmp/test_index")
    store.build(chunks, embeddings)

    hasil = store.search([1.0, 0.0, 0.0], top_k=2, active_only=True)
    assert len(hasil) >= 1
    chunk_teratas, skor = hasil[0]
    assert chunk_teratas.chunk_id == "c1"
    assert skor > 0.9


def test_search_hanya_ambil_yang_aktif():
    chunks = [
        buat_chunk_dummy("aktif1", "teks aktif"),
        buat_chunk_dummy("arsip1", "teks arsip", is_active=False, doc_version="1.4"),
    ]
    embeddings = [
        [1.0, 0.0],
        [1.0, 0.0],
    ]
    store = VectorStore(index_dir="/tmp/test_index2")
    store.build(chunks, embeddings)

    hasil = store.search([1.0, 0.0], top_k=5, active_only=True)
    ids = [c.chunk_id for c, s in hasil]
    assert "arsip1" not in ids
