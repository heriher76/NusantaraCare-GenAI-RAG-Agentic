import re
from dataclasses import dataclass
from typing import Optional

INJECTION_PATTERNS = [
    r"abaikan (semua )?instruksi",
    r"ignore (all )?(previous|prior|above) instructions",
    r"lupakan (aturan|instruksi|peranmu)",
    r"kamu (sekarang|adalah) (AI|asisten) tanpa batasan",
    r"you are now (in )?(dan|developer)?\s*mode",
    r"tampilkan (system prompt|prompt sistem|api key|kredensial)",
    r"reveal (the )?(system prompt|api key)",
    r"bertindak sebagai (root|admin|developer) (tanpa )?filter",
    r"disregard (the )?(rules|guidelines|policy)",
    r"pretend (you are|to be)",
    r"jailbreak",
]

INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

OUT_OF_SCOPE_KEYWORDS = [
    "gaji", "tunjangan", "kompensasi", "bonus", "payroll",
    "konsultasi medis", "sakit", "dokter", "kesehatan karyawan",
    "nasihat hukum", "hukum", "somasi", "gugatan",
    "penilaian kinerja", "konseling kinerja", "performance review",
]


@dataclass
class GuardResult:
    blocked: bool
    reason: Optional[str] = None


def check_prompt_injection(question):
    if INJECTION_RE.search(question):
        return GuardResult(blocked=True, reason="blocked_prompt_injection")
    return GuardResult(blocked=False)


def check_out_of_scope_hint(question):
    lowered = question.lower()
    for kw in OUT_OF_SCOPE_KEYWORDS:
        if kw in lowered:
            return GuardResult(blocked=False, reason="possible_out_of_scope")
    return GuardResult(blocked=False)
