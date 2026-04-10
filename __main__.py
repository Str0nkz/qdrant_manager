"""Package entry point for the Qdrant cleaner."""

from __future__ import annotations

from .cleaner_cli import QdrantCleanerCLI


def main() -> None:
    """Run the cleaner CLI."""
    QdrantCleanerCLI().run()


if __name__ == "__main__":
    main()
