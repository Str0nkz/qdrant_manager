"""Public package surface for the Qdrant Manager."""
from .manager import QdrantManager
from .manager_cli import QdrantManagerCLI

__all__ = ["QdrantManager", "QdrantManagerCLI"]
