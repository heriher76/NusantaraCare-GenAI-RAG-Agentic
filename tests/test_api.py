from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@patch("app.main.answer_question")
def test_ask_endpoint(mock_answer_question):
    from app.schemas import AskResponse

    mock_answer_question.return_value = AskResponse(
        answer="Tidak ditemukan dalam dokumen.",
        confidence_label="low",
        reason_code="no_relevant_context",
        sources=[],
    )

    resp = client.post("/ask", json={"question": "Pertanyaan random"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reason_code"] == "no_relevant_context"


def test_ask_endpoint_pertanyaan_kosong_ditolak():
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422
