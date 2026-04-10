"""Open-source configuration for the standalone Qdrant cleaner."""
from __future__ import annotations

import os

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_DIRECT = QDRANT_URL
CANONICAL_TEXT_FIELDS = ["text", "content", "information"]
CANONICAL_ROOT_FIELDS: list[str] = []
