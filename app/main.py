from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.schemas import AskRequest, AskResponse, HealthResponse
from app.services.rag import answer_question, ensure_index_ready
from app.services.vectorstore import vector_store

app = FastAPI(
    title=config.APP_NAME,
    description="Asisten GenAI berbasis RAG untuk dokumen internal NusantaraCare.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    try:
        ensure_index_ready()
    except Exception as exc:
        print(f"[startup] gagal membangun index: {exc}")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        index_loaded=vector_store.index is not None,
        chunk_count=len(vector_store.chunks),
    )


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    try:
        return answer_question(payload.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"gagal memproses pertanyaan: {exc}")
