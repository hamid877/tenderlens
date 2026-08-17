"""Shared pytest fixtures for TenderLens tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.document_service import document_service


@pytest.fixture(autouse=True, scope="session")
def _redirect_upload_dir(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point the document_service singleton at a temporary directory.

    This fixture runs automatically for every test session and ensures that no
    test-generated PDF files are written into the real ``data/uploads/``
    directory.  The temporary directory is managed by pytest and removed after
    the session ends.

    Production behavior is unchanged: when the application runs normally the
    singleton is constructed with ``upload_dir=None``, which resolves to
    ``data/uploads/`` as before.
    """
    tmp_uploads: Path = tmp_path_factory.mktemp("uploads")
    document_service._upload_dir = tmp_uploads
