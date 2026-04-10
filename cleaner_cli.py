"""Lean interactive CLI for the open-source Qdrant cleaner."""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from cleaner import QdrantCleaner
from config import QDRANT_DIRECT, QDRANT_API_KEY


_COLOUR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text


def bold(text: str) -> str:
    return _c("1", text)


def dim(text: str) -> str:
    return _c("2", text)


def red(text: str) -> str:
    return _c("91", text)


def green(text: str) -> str:
    return _c("92", text)


def yellow(text: str) -> str:
    return _c("93", text)


def cyan(text: str) -> str:
    return _c("96", text)


def ok(message: str) -> None:
    print(f"  {green('✓')} {message}")


def warn(message: str) -> None:
    print(f"  {yellow('⚠')} {message}")


def fail(message: str) -> None:
    print(f"  {red('✗')} {message}")


def _hr(width: int = 62, ch: str = "─") -> None:
    print(f"  {dim(ch * width)}")


class QdrantCleanerCLI:
    """Cleaner-focused Qdrant Manager CLI"""

    MENU = [
        ("1", "List Collections"),
        ("2", "Analyze Collection"),
        ("3", "Clean Collection"),
        ("4", "Repair Vectors"),
        ("5", "Semantic Deduplication"),
        ("6", "Scan Sparse Content"),
        ("7", "Outlier Detection"),
        ("8", "Remove Points by Payload Filter"),
        ("9", "Set Qdrant URL"),
    ]

    def __init__(self, *, qdrant_url: str = QDRANT_DIRECT) -> None:
        self.qdrant_url = ""
        self.cleaner: Optional[QdrantCleaner] = None
        self.collections: List[Dict[str, Any]] = []
        self._set_qdrant_url(qdrant_url)

    def _set_qdrant_url(self, qdrant_url: str, api_key: str = None) -> None:
        qdrant_url = (qdrant_url or "").strip() or QDRANT_DIRECT
        parsed = urlparse(qdrant_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6333
        api_key = api_key or QDRANT_API_KEY
        self.qdrant_url = qdrant_url
        self.cleaner = QdrantCleaner(host=host, port=port, api_key=api_key)
        self.collections = []

    def _progress(self, message: str) -> None:
        message = message.strip()
        if message:
            print(f"  {dim(message)}")

    def _prompt(self, message: str, default: Optional[str] = None) -> str:
        suffix = f" {dim(f'[{default}]')}" if default is not None else ""
        value = input(f"  {yellow('?')} {message}{suffix}: ").strip()
        return value if value else (default or "")

    def _confirm(self, message: str, default: bool = True) -> bool:
        options = "Y/n" if default else "y/N"
        value = input(f"  {yellow('?')} {message} [{options}]: ").strip().lower()
        if not value:
            return default
        return value in {"y", "yes"}

    def _choose_collection(self, prompt: str) -> Optional[str]:
        if not self.collections:
            self.load_collections()
        if not self.collections:
            warn("No collections available.")
            return None

        print(f"\n  {bold('Available collections')}")
        _hr(56)
        for idx, coll in enumerate(self.collections, start=1):
            points_label = f"({coll['points']:,} pts)"
            print(f"  {cyan(str(idx).rjust(2))}  {coll['name']:<32} {dim(points_label)}")
        _hr(56)

        raw = input(f"  {yellow('?')} {prompt} [1-{len(self.collections)}]: ").strip()
        if not raw:
            return None
        try:
            index = int(raw) - 1
        except ValueError:
            warn("Enter a valid number.")
            return None
        if 0 <= index < len(self.collections):
            return self.collections[index]["name"]
        warn("Choice out of range.")
        return None

    def _check_connection(self) -> bool:
        """Ping Qdrant and print a green/red status line. Returns True if reachable."""
        if self.cleaner is None:
            self._set_qdrant_url(self.qdrant_url or QDRANT_DIRECT)
        if self.cleaner.ping():
            ok(f"Connected  {dim(self.qdrant_url)}")
            return True
        fail(f"Unreachable  {dim(self.qdrant_url)}")
        return False

    def load_collections(self) -> bool:
        """Load collections. Returns True on success, False on connection failure."""
        if self.cleaner is None:
            self._set_qdrant_url(self.qdrant_url or QDRANT_DIRECT)
        try:
            self.collections = self.cleaner.list_collections()
            return True
        except Exception as exc:
            warn(f"Could not load collections: {exc}")
            self.collections = []
            return False

    def _configure_qdrant_url(self) -> None:
        new_url = self._prompt("Qdrant URL", default=self.qdrant_url or QDRANT_DIRECT)
        self._set_qdrant_url(new_url)
        if self._check_connection():
            self.load_collections()

    def _print_header(self) -> None:
        title = "QDRANT CLEANER"
        width = 62
        print(f"  {dim('═' * width)}")
        print(f"  {bold(cyan(title.center(width)))}")
        print(f"  {dim('═' * width)}")
        print(f"  {dim('Endpoint:')} {cyan(self.qdrant_url)}")
        _hr(width)
        print(f"  {dim('Cleaner-only OSS mode: no proxy, ingest, migrate, or refresh actions exposed.')}")

    def _print_menu(self) -> None:
        self._print_header()
        print()
        for key, label in self.MENU:
            print(f"    {cyan(key)}  {label}")
        print()
        _hr(62)
        print(f"    {cyan('0')}  Exit")
        print()

    def _select_collection(self, prompt: str) -> Optional[str]:
        selection = self._choose_collection(prompt)
        if selection:
            if self.cleaner is None:
                self._set_qdrant_url(self.qdrant_url or QDRANT_DIRECT)
            self.cleaner.set_collection(selection)
        return selection

    def _list_collections(self) -> None:
        self.load_collections()
        if not self.collections:
            warn("No collections found.")
            return
        print()
        print(f"  {bold('Name'):<36} {bold('Points'):<12} {bold('Vector'):<16} {bold('Segments')}")
        _hr(72)
        for coll in self.collections:
            print(
                f"  {cyan(coll['name']):<36} "
                f"{str(coll['points']):<12} "
                f"{coll['vector_name']:<16} "
                f"{coll['segments']}"
            )

    def _analyze_collection(self) -> None:
        collection = self._select_collection("Collection to analyze")
        if not collection:
            return
        result = self.cleaner.analyze_collection(progress_callback=self._progress)
        print()
        ok(f"Health score: {result.get('health_score', 0)}/100")
        print(f"  Duplicate points : {result.get('duplicate_count', 0):,}")
        print(f"  Empty vectors    : {len(result.get('empty_vectors', [])):,}")
        print(f"  Missing text     : {len(result.get('missing_text', [])):,}")
        for issue in result.get("issues", [])[:8]:
            warn(issue)

    def _clean_collection(self) -> None:
        collection = self._select_collection("Collection to clean")
        if not collection:
            return
        dry_run = self._confirm("Dry run only?", default=True)
        keep_per_group = int(self._prompt("Keep per duplicate group", default="1"))
        optimize = self._confirm("Optimize after cleanup?", default=False)
        result = self.cleaner.one_stop_clean(
            dry_run=dry_run,
            keep_per_group=keep_per_group,
            optimize=optimize,
            progress_callback=self._progress,
        )
        print()
        if dry_run:
            warn(f"Dry run: {result.get('removed', 0)} duplicate points would be removed")
        else:
            ok(f"Removed {result.get('removed', 0)} duplicate points")

    def _repair_vectors(self) -> None:
        collection = self._select_collection("Collection to repair")
        if not collection:
            return
        normalize = self._confirm("Normalize vectors?", default=True)
        remove_invalid = self._confirm("Remove invalid vectors?", default=True)
        dry_run = self._confirm("Dry run only?", default=True)
        result = self.cleaner.repair_vectors(
            normalize=normalize,
            remove_invalid=remove_invalid,
            dry_run=dry_run,
            progress_callback=self._progress,
        )
        print()
        ok(
            "Scanned {scanned:,}, removed {removed:,}, normalized {repaired:,}".format(
                scanned=result.get("scanned", 0),
                removed=result.get("removed", 0),
                repaired=result.get("repaired", 0),
            )
        )

    def _semantic_dedup(self) -> None:
        collection = self._select_collection("Collection for semantic deduplication")
        if not collection:
            return
        threshold = float(self._prompt("Similarity threshold", default="0.98"))
        strategy = self._prompt("Keep strategy [newest|oldest|most_complete]", default="newest")
        use_clustering = self._confirm("Use clustering?", default=True)
        cluster_backend = "auto"
        if use_clustering:
            cluster_backend = self._prompt("Cluster backend [auto|gpu|cpu]", default="auto")
        dry_run = self._confirm("Dry run only?", default=True)
        result = self.cleaner.semantic_deduplication(
            similarity_threshold=threshold,
            keep_strategy=strategy,
            use_clustering=use_clustering,
            cluster_backend=cluster_backend,
            dry_run=dry_run,
            progress_callback=self._progress,
        )
        print()
        ok(
            "Groups: {groups:,}  Duplicates: {dupes:,}  Deleted: {deleted:,}".format(
                groups=result.get("duplicate_groups", 0),
                dupes=result.get("total_duplicates", 0),
                deleted=result.get("deleted", 0),
            )
        )

    def _scan_sparse_content(self) -> None:
        collection = self._select_collection("Collection to scan")
        if not collection:
            return
        percentile = float(self._prompt("Bottom percentile", default="0.10"))
        vector_weight = float(self._prompt("Vector anomaly weight", default="0.55"))
        action = self._prompt("Action [report|tag|delete]", default="report")
        dry_run = True if action == "report" else self._confirm("Dry run only?", default=True)
        result = self.cleaner.scan_sparse(
            percentile=percentile,
            vector_weight=vector_weight,
            action=action,
            dry_run=dry_run,
            progress_callback=self._progress,
        )
        print()
        ok(f"Candidates: {result.get('candidates', 0):,}")
        if action == "tag":
            print(f"  Tagged  : {result.get('tagged', 0):,}")
        if action == "delete":
            print(f"  Deleted : {result.get('deleted', 0):,}")

    def _detect_outliers(self) -> None:
        collection = self._select_collection("Collection for outlier detection")
        if not collection:
            return
        method = self._prompt("Method [isolation_forest|dbscan|statistical]", default="isolation_forest")
        contamination = float(self._prompt("Contamination", default="0.01"))
        cluster_backend = "auto"
        if method == "dbscan":
            cluster_backend = self._prompt("Cluster backend [auto|gpu|cpu]", default="auto")
        remove = self._confirm("Remove outliers?", default=False)
        dry_run = self._confirm("Dry run only?", default=True)
        result = self.cleaner.detect_outliers(
            method=method,
            contamination=contamination,
            cluster_backend=cluster_backend,
            remove=remove,
            dry_run=dry_run,
            progress_callback=self._progress,
        )
        print()
        ok(
            "Vectors: {total:,}  Outliers: {outliers:,}  Removed: {removed:,}".format(
                total=result.get("total_vectors", 0),
                outliers=result.get("outliers_detected", 0),
                removed=result.get("removed", 0),
            )
        )

    def _remove_points_by_filter(self) -> None:
        collection = self._select_collection("Collection to clean")
        if not collection:
            return
        print(f"\n  {dim('Enter payload field=value filters, one per line. Blank line to finish.')}")
        filters = {}
        while True:
            entry = input(f"  {yellow('?')} field=value: ").strip()
            if not entry:
                break
            if "=" not in entry:
                warn("Expected field=value format.")
                continue
            key, _, val = entry.partition("=")
            filters[key.strip()] = val.strip()
        if not filters:
            warn("No filters entered.")
            return
        dry_run = self._confirm("Dry run only?", default=True)
        result = self.cleaner.remove_points_by_payload_filter(
            filters=filters,
            dry_run=dry_run,
            progress_callback=self._progress,
        )
        print()
        ok(
            "Found: {found:,}  Removed: {removed:,}".format(
                found=result.get("total_found", 0),
                removed=result.get("removed", 0),
            )
        )

    def _startup(self) -> None:
        """Prompt for Qdrant URL, check connectivity, and load collections."""
        width = 62
        print(f"\n  {dim('═' * width)}")
        print(f"  {bold(cyan('QDRANT CLEANER'.center(width)))}")
        print(f"  {dim('═' * width)}")
        print()

        new_url = self._prompt("Qdrant URL", default=self.qdrant_url or QDRANT_DIRECT)
        if new_url != self.qdrant_url:
            self._set_qdrant_url(new_url)

        while True:
            reachable = self._check_connection()
            if reachable:
                self.load_collections()
                if self.collections:
                    ok(f"{len(self.collections)} collection(s) found")
                else:
                    warn("No collections found")
                break
            try:
                retry = self._confirm("Try a different URL?", default=True)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not retry:
                warn("Continuing offline — collection actions will fail until Qdrant is reachable.")
                break
            new_url = self._prompt("Qdrant URL", default=self.qdrant_url)
            self._set_qdrant_url(new_url)

        print()

    def run(self) -> None:
        dispatch = {
            "1": self._list_collections,
            "2": self._analyze_collection,
            "3": self._clean_collection,
            "4": self._repair_vectors,
            "5": self._semantic_dedup,
            "6": self._scan_sparse_content,
            "7": self._detect_outliers,
            "8": self._remove_points_by_filter,
            "9": self._configure_qdrant_url,
        }

        self._startup()
        while True:
            self._print_menu()
            try:
                choice = input(f"  {yellow('›')} Choice: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if choice in {"0", "q", "quit", "exit"}:
                print()
                break

            action = dispatch.get(choice)
            if action is None:
                warn(f"Unknown choice: {choice}")
            else:
                try:
                    print()
                    action()
                except KeyboardInterrupt:
                    warn("Cancelled.")
                except Exception as exc:
                    fail(str(exc))

            input(f"\n  {dim('Press Enter to continue...')}")


def main() -> None:
    """Standalone entry point for the OSS cleaner CLI."""
    QdrantCleanerCLI().run()


if __name__ == "__main__":
    main()
