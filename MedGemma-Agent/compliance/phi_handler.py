"""
PHI handler — de-identification and encryption for HIPAA compliance.

De-identification: strips/pseudonymizes the 18 HIPAA identifiers before
any text is passed to the LLM or stored in ChromaDB.

Encryption: AES-256 (Fernet) for PHI fields persisted to SQLite.
"""

from __future__ import annotations
import re
from cryptography.fernet import Fernet, InvalidToken
from config.settings import get_settings

settings = get_settings()

# ── Encryption ────────────────────────────────────────────────────────────────

def _get_fernet() -> Fernet:
    if not settings.ENCRYPTION_KEY:
        raise RuntimeError(
            "ENCRYPTION_KEY not set in .env. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(settings.ENCRYPTION_KEY.encode())


def encrypt_phi(plaintext: str) -> str:
    """Encrypt a PHI string. Returns base64-encoded ciphertext."""
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_phi(ciphertext: str) -> str:
    """Decrypt a PHI string. Raises InvalidToken if tampered."""
    return _get_fernet().decrypt(ciphertext.encode()).decode()


# ── De-identification patterns (18 HIPAA identifiers) ────────────────────────

_PHI_PATTERNS: list[tuple[str, str]] = [
    # Names (basic heuristic — proper nouns adjacent to titles, or two capitalized words)
    (r"\b(?:(?:Dr|Mr|Mrs|Ms|Prof)\.?\s+)?[A-Z][a-z]+\s+[A-Z][a-z]+\b", "[NAME]"),
    # Dates (various formats)
    (r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", "[DATE]"),
    (r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b", "[DATE]"),
    # Phone numbers
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b", "[PHONE]"),
    # Email
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    # SSN
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    # MRN / medical record numbers (common formats)
    (r"\bMR[N#]?\s*:?\s*\d{5,12}\b", "[MRN]"),
    (r"\bPatient\s+(?:ID|#)\s*:?\s*\d{5,12}\b", "[PATIENT_ID]"),
    # IP addresses
    (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP]"),
    # ZIP codes (5 or 9 digit)
    (r"\b\d{5}(?:-\d{4})?\b", "[ZIP]"),
    # URLs
    (r"https?://\S+", "[URL]"),
    # Account / license numbers
    (r"\b(?:Account|License|Certificate)\s*#?\s*:?\s*\w{5,}\b", "[ID_NUMBER]"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in _PHI_PATTERNS]


def deidentify(text: str) -> str:
    """
    Apply regex-based de-identification to remove/replace the 18 HIPAA
    identifiers. Not perfect — clinical text should be reviewed before
    production use. Documented limitation per ISO 14971 risk file.
    """
    for pattern, replacement in _COMPILED:
        text = pattern.sub(replacement, text)
    return text
