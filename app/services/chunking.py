import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any

import yaml

from app import config


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    doc_title: str
    doc_version: str
    is_active: bool
    section_title: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    line_start: int = 0
    line_end: int = 0


def parse_frontmatter(raw):
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.DOTALL)
    if not match:
        return {}, raw
    fm_raw, body = match.groups()
    fm = yaml.safe_load(fm_raw) or {}
    return fm, body


def split_by_headings(body):
    lines = body.split("\n")
    sections = []
    current_title = "Pendahuluan"
    current_lines = []
    current_start = 1

    for i, line in enumerate(lines, start=1):
        heading_match = re.match(r"^(#{2,3})\s+(.*)$", line.strip())
        if heading_match:
            if current_lines:
                sections.append({
                    "title": current_title,
                    "text": "\n".join(current_lines).strip(),
                    "line_start": current_start,
                    "line_end": i - 1,
                })
            current_title = heading_match.group(2).strip()
            current_lines = [line]
            current_start = i
        else:
            current_lines.append(line)
    if current_lines:
        sections.append({
            "title": current_title,
            "text": "\n".join(current_lines).strip(),
            "line_start": current_start,
            "line_end": len(lines),
        })
    return [s for s in sections if s["text"]]


def detect_version_flags(text, doc_default_active, doc_default_version):
    is_v14_archive = bool(re.search(r"v1\.4", text)) and bool(
        re.search(r"NONAKTIF|tidak aktif|tidak berlaku|is_active:\s*false", text, re.IGNORECASE)
    )
    if is_v14_archive:
        return {"doc_version": "1.4", "is_active": False}
    return {"doc_version": doc_default_version, "is_active": doc_default_active}


def _paragraphs_with_lines(section_text, section_line_start):
    section_lines = section_text.split("\n")
    paragraphs = []
    buf_lines = []
    buf_start = 0

    def flush(end_idx):
        if buf_lines:
            text = "\n".join(buf_lines).strip()
            if text:
                paragraphs.append({
                    "text": text,
                    "line_start": section_line_start + buf_start,
                    "line_end": section_line_start + end_idx,
                })

    idx = 0
    for i, line in enumerate(section_lines):
        if line.strip() == "":
            flush(i - 1)
            buf_lines = []
            buf_start = i + 1
        else:
            buf_lines.append(line)
        idx = i
    flush(idx)
    return paragraphs


def chunk_text(text, chunk_size, overlap):
    paragraphs = [{"text": p.strip(), "line_start": 0, "line_end": 0} for p in text.split("\n\n") if p.strip()]
    chunked = chunk_text_with_lines(paragraphs, chunk_size, overlap)
    return [c["text"] for c in chunked]


def chunk_text_with_lines(paragraphs_with_lines, chunk_size, overlap):
    chunks = []
    current_text = ""
    current_line_start = None
    current_line_end = None

    def push_current():
        if current_text:
            chunks.append({
                "text": current_text,
                "line_start": current_line_start,
                "line_end": current_line_end,
            })

    for para in paragraphs_with_lines:
        para_text = para["text"]
        if len(current_text) + len(para_text) + 2 <= chunk_size:
            current_text = f"{current_text}\n\n{para_text}".strip()
            current_line_start = para["line_start"] if current_line_start is None else min(current_line_start, para["line_start"])
            current_line_end = para["line_end"] if current_line_end is None else max(current_line_end, para["line_end"])
        else:
            push_current()
            if len(para_text) <= chunk_size:
                current_text = para_text
                current_line_start = para["line_start"]
                current_line_end = para["line_end"]
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para_text)
                buf = ""
                for sent in sentences:
                    if len(buf) + len(sent) + 1 <= chunk_size:
                        buf = f"{buf} {sent}".strip()
                    else:
                        if buf:
                            chunks.append({
                                "text": buf,
                                "line_start": para["line_start"],
                                "line_end": para["line_end"],
                            })
                        buf = sent
                if buf:
                    chunks.append({
                        "text": buf,
                        "line_start": para["line_start"],
                        "line_end": para["line_end"],
                    })
                current_text = ""
                current_line_start = None
                current_line_end = None

    push_current()

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1]["text"][-overlap:]
            overlapped.append({
                "text": f"{tail}\n\n{chunks[i]['text']}",
                # line_start ikut chunk sebelumnya karena tail-nya diambil dari sana
                "line_start": min(chunks[i - 1]["line_start"], chunks[i]["line_start"]),
                "line_end": chunks[i]["line_end"],
            })
        return overlapped
    return chunks


def load_and_chunk_document(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    frontmatter, body = parse_frontmatter(raw)
    doc_id = frontmatter.get("doc_id", "UNKNOWN")
    doc_title = frontmatter.get("doc_title", "Dokumen Tanpa Judul")
    doc_version = str(frontmatter.get("doc_version", "unknown"))
    is_active = bool(frontmatter.get("is_active", True))

    sections = split_by_headings(body)
    all_chunks = []

    for section in sections:
        version_flags = detect_version_flags(section["text"], is_active, doc_version)
        paragraphs = _paragraphs_with_lines(section["text"], section["line_start"])
        raw_chunks = chunk_text_with_lines(paragraphs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for raw_chunk in raw_chunks:
            all_chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4())[:8],
                    text=raw_chunk["text"],
                    doc_id=doc_id,
                    doc_title=doc_title,
                    doc_version=version_flags["doc_version"],
                    is_active=version_flags["is_active"],
                    section_title=section["title"],
                    metadata={
                        "source_path": frontmatter.get("source_path", path),
                        "owner": frontmatter.get("owner", ""),
                        "effective_date": frontmatter.get("effective_date", ""),
                    },
                    line_start=raw_chunk["line_start"] or section["line_start"],
                    line_end=raw_chunk["line_end"] or section["line_end"],
                )
            )
    return all_chunks