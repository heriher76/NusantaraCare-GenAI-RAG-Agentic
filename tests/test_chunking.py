from app.services.chunking import load_and_chunk_document, chunk_text, detect_version_flags


def test_load_and_chunk_document():
    chunks = load_and_chunk_document("data/raw_docs/nusantaracare_panduan_operasional_internal_v2.md")
    assert len(chunks) > 0


def test_ada_chunk_arsip_v14():
    chunks = load_and_chunk_document("data/raw_docs/nusantaracare_panduan_operasional_internal_v2.md")
    archived = [c for c in chunks if not c.is_active]
    assert len(archived) >= 1
    assert archived[0].doc_version == "1.4"


def test_mayoritas_chunk_aktif_v2():
    chunks = load_and_chunk_document("data/raw_docs/nusantaracare_panduan_operasional_internal_v2.md")
    active = [c for c in chunks if c.is_active]
    assert len(active) > len(chunks) / 2
    assert active[0].doc_version == "2.0"


def test_chunk_text_tidak_melebihi_batas_wajar():
    text = "kalimat satu. " * 200
    hasil = chunk_text(text, chunk_size=100, overlap=0)
    assert len(hasil) > 1


def test_detect_version_flags_arsip():
    teks = "Versi v1.4 kini berstatus NONAKTIF dan tidak boleh digunakan."
    hasil = detect_version_flags(teks, True, "2.0")
    assert hasil["is_active"] is False
    assert hasil["doc_version"] == "1.4"


def test_detect_version_flags_aktif():
    teks = "Ini bagian biasa tentang SOP akses."
    hasil = detect_version_flags(teks, True, "2.0")
    assert hasil["is_active"] is True
    assert hasil["doc_version"] == "2.0"
