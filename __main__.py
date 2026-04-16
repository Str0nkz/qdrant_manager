"""Package entry point for the Qdrant Manager."""

from __future__ import annotations

from .manager_cli import QdrantManagerCLI


def main() -> None:
    """Run the manager CLI."""
    QdrantManagerCLI().run()


if __name__ == "__main__":
    main()
