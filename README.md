NusantaraCare RAG Assistant

Project ini adalah backend tanya jawab berbasis RAG untuk membantu karyawan mencari informasi dari Panduan Operasional Layanan Internal NusantaraCare v2.0. Backend dibuat menggunakan FastAPI.

Embedding dan chat menggunakan Notispace (api.notispaces.cloud), bukan OpenAI atau Anthropic. Format endpoint chat diasumsikan mirip OpenAI (/v1/chat/completions). Jika format aslinya berbeda, penyesuaian bisa dilakukan di app/services/notispace_client.py.

Tujuan

Sistem ini dibuat supaya karyawan bisa mendapatkan jawaban dari dokumen internal dengan lebih cepat. Sistem dirancang untuk:

Menjawab berdasarkan dokumen resmi saja.

Menyertakan sumber jawaban, seperti versi dokumen, bagian, dan chunk.

Menolak pertanyaan yang tidak dibahas di dokumen.

Memblokir pola prompt injection dari pengguna.

Membedakan aturan aktif v2.0 dan arsip v1.4.

Pertanyaan yang ditargetkan mencakup akses dan akun, gangguan layanan, insiden P1, perlengkapan kerja, keamanan informasi, kerahasiaan tiket, dan perubahan aturan dari v1.4 ke v2.0.

Saat ini sistem hanya menggunakan satu knowledge base: nusantaracare_panduan_operasional_internal_v2.md. Topik seperti medis, hukum, gaji, penilaian kinerja, dan infrastruktur di luar Direktorat Operasi dan Layanan Internal berada di luar cakupan.

Knowledge Base

Dokumen utama memiliki metadata berikut:

doc_id: NC-OPS-001

doc_title: Panduan Operasional Layanan Internal NusantaraCare

doc_version: 2.0

effective_date: 2026-07-01

last_updated: 2026-07-15

is_active: true

owner: Direktorat Operasi dan Layanan Internal

Dokumen berisi tujuan dan ruang lingkup, istilah dan peran, kanal layanan, SOP, FAQ, matriks keputusan, serta riwayat perubahan.

Perbedaan aturan yang perlu diperhatikan:

Ketentuan

v1.4 (arsip)

v2.0 (aktif)

Pengajuan lewat email

Email biasa setara dengan portal

Email hanya untuk keadaan darurat saat portal tidak tersedia, dengan tanda [DARURAT-PORTAL]

Pengajuan perlengkapan

Minimal 3 hari kerja

Minimal 5 hari kerja

Versi v1.4 tidak berlaku sejak 1 Juli 2026. Karena kedua versi ada di satu file, status versi ditentukan pada tiap chunk.

RAG Design

Dokumen dipecah berdasarkan heading terlebih dahulu, lalu paragraf. Target ukuran chunk adalah 900 karakter dengan overlap 150 karakter. Jika ada paragraf yang terlalu panjang, sistem memecahnya per kalimat.

Setiap chunk menyimpan ID chunk, ID dan judul dokumen, versi, status aktif, judul bagian, pemilik, tanggal berlaku, dan lokasi sumber.

Fungsi _detect_version_flags menandai bagian arsip v1.4 sebagai doc_version: "1.4" dan is_active: false. Dari 96 chunk, satu chunk arsip ditandai nonaktif dan sisanya aktif.

FAISS digunakan sebagai vector database lokal. Sistem memakai IndexFlatIP dengan embedding yang dinormalisasi untuk pencarian cosine similarity. Index disimpan di data/index/.

Pengaturan retrieval:

TOP_K: jumlah hasil utama, default 5.

SIMILARITY_THRESHOLD: batas relevansi, default 0.35.

Pencarian mengutamakan chunk aktif. Chunk arsip digunakan jika tidak ada chunk aktif yang cukup relevan.

Prompt menginstruksikan model untuk menjawab hanya dari konteks, menyatakan jika jawaban tidak ditemukan, menjelaskan jika sumbernya hanya arsip, dan mengabaikan instruksi yang muncul di dokumen atau pertanyaan pengguna. Jawaban dikembalikan dalam JSON dengan field answer, confidence_label, dan reason_code.

Keamanan

Sebelum memanggil model, sistem memeriksa pola prompt injection menggunakan regex. Contohnya adalah instruksi untuk mengabaikan arahan sebelumnya atau menampilkan system prompt.

System prompt juga meminta model mengabaikan instruksi yang tertanam di dokumen. Jika hasil pencarian tidak mencapai threshold, sistem menyatakan bahwa informasi tidak ditemukan dalam dokumen.

Alur Sistem

Pengguna mengirim pertanyaan ke endpoint /ask.

Sistem memeriksa prompt injection.

Sistem memuat atau membangun index FAISS.

Pertanyaan diubah menjadi embedding melalui Notispace.

FAISS mencari chunk yang relevan.

Sistem memeriksa threshold dan status versi.

Prompt disusun dari konteks dan pertanyaan.

Notispace menghasilkan jawaban JSON beserta sumber.

Struktur Repository

repository/
  README.md
  data/
    raw_docs/
      nusantaracare_panduan_operasional_internal_v2.md
  app/
    main.py
    config.py
    schemas.py
    services/
      notispace_client.py
      chunking.py
      vectorstore.py
      agent.py
      rag.py
  tests/
    test_chunking.py
    test_agent.py
    test_vectorstore.py
    test_rag.py
    test_api.py
  requirements.txt
  .env.example
  .gitignore

API

POST /ask

Contoh request:

{
  "question": "Berapa lama tenggat pengajuan perlengkapan standar?"
}

Contoh response:

{
  "answer": "Permintaan perlengkapan standar wajib diajukan minimal 5 hari kerja sebelum tanggal kebutuhan, disertai persetujuan Atasan Langsung.",
  "confidence_label": "high",
  "reason_code": "answered",
  "sources": [
    {
      "doc_id": "NC-OPS-001",
      "doc_title": "Panduan Operasional Layanan Internal NusantaraCare",
      "doc_version": "2.0",
      "is_active": true,
      "section_title": "Pertanyaan Perlengkapan",
      "chunk_id": "a1b2c3d4"
    }
  ]
}

Nilai reason_code yang tersedia: answered, no_relevant_context, out_of_scope, blocked_prompt_injection, dan archived_version_only.

GET /health

Contoh response:

{
  "status": "ok",
  "index_loaded": true,
  "chunk_count": 96
}

How to menjalankan Lokal

python3 -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env

Isi NOTISPACE_API_KEY di file .env, lalu jalankan:

uvicorn app.main:app --reload --port 8000

Coba endpoint:

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Apakah email biasa masih bisa dipakai untuk mengajukan permintaan?"}'

How to menjalankan Test

Test menggunakan pytest dan mencakup chunking, prompt injection, vector store, pipeline RAG dengan mock Notispace, dan endpoint API.

pip install -r requirements.txt
pytest -v

Deployment

Push repository ke GitHub.

Deploy ke FastAPI Cloud dengan entrypoint app.main:app.

Atur NOTISPACE_API_KEY sebagai secret.

Jika perlu, atur NOTISPACE_BASE_URL, NOTISPACE_EMBED_MODEL, dan NOTISPACE_CHAT_MODEL.

Index FAISS dibuat saat startup pertama dan disimpan di data/index/.

Limitasi dan Pengembangan

Deteksi arsip v1.4 bergantung pada pola teks, sehingga perlu disesuaikan untuk dokumen lain.

Belum ada evaluasi otomatis dengan kumpulan pertanyaan dan metrik retrieval.

Regex prompt injection tidak bisa menangkap semua variasi instruksi.

Pengembangan berikutnya bisa mencakup evaluasi retrieval, classifier prompt injection, dan dukungan beberapa dokumen.