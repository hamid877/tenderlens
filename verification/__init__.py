"""Verification package – claim extraction and fact-checking utilities."""

from verification.claim_extractor import Claim, ClaimExtractor
from verification.evidence_verifier import (
    DEFAULT_MODEL_NAME,
    EvidenceVerifier,
    MultiEvidenceVerificationResult,
    VerificationResult,
)

__all__ = [
    "Claim",
    "ClaimExtractor",
    "DEFAULT_MODEL_NAME",
    "EvidenceVerifier",
    "MultiEvidenceVerificationResult",
    "VerificationResult",
]
