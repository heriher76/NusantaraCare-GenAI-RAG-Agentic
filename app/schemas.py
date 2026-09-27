from typing import List, Literal
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceCitation(BaseModel):
    doc_id: str
    doc_title: str
    doc_version: str
    is_active: bool
    section_title: str
    chunk_id: str
    line_start: int = 0
    line_end: int = 0


class AskResponse(BaseModel):
    answer: str
    confidence_label: Literal["high", "medium", "low"]
    reason_code: Literal[
        "answered",
        "no_relevant_context",
        "out_of_scope",
        "blocked_prompt_injection",
        "archived_version_only",
    ]
    sources: List[SourceCitation] = []


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    chunk_count: int
