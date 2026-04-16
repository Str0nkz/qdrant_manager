"""Utilities for managing and repairing Qdrant collections."""
from __future__ import annotations

import hashlib
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .config import CANONICAL_TEXT_FIELDS
from .dependencies import (
    CuMLDBSCAN,
    GPU_CLUSTERING_AVAILABLE,
    IsolationForest,
    NUMPY_AVAILABLE,
    OptimizersConfigDiff,
    PointIdsList,
    QdrantClient,
    PCA,
    SKLEARN_AVAILABLE,
    SklearnDBSCAN,
    cp,
    np,
    pd,
)


class QdrantManager:
    """Collection cleaning utilities"""

    def __init__(self, host="localhost", port=6333):
        if QdrantClient is None:
            raise ImportError("qdrant-client required")
        
        if host.startswith("http://") or host.startswith("https://"):
            parsed = urlparse(host)
            host = parsed.hostname
            port = parsed.port or port
        
        self.client = QdrantClient(host=host, port=int(port))
        self.collection_name = None
        self.has_optimize_method = hasattr(self.client, 'optimize_collection')

    def set_collection(self, collection_name: str):
        self.collection_name = collection_name

    def ping(self) -> bool:
        """Return True if the Qdrant instance is reachable."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def list_collections(self, *, skip_collections: Optional[set[str]] = None) -> List[Dict[str, Any]]:
        """List Qdrant collections with lightweight metadata."""
        skip = skip_collections or set()
        collections = []

        for coll in self.client.get_collections().collections:
            if coll.name in skip:
                continue

            info = self.client.get_collection(coll.name)
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict) and "size" in vectors_config:
                vector_name = "default"
            elif isinstance(vectors_config, dict):
                vector_name = next(iter(vectors_config.keys()), "unknown")
            else:
                vector_name = "unknown"

            collections.append(
                {
                    "name": coll.name,
                    "points": info.points_count,
                    "vector_name": vector_name,
                    "segments": info.segments_count,
                }
            )

        return collections

    def _parse_timestamp(self, value: Any) -> float:
        """Parse common timestamp formats to epoch seconds."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            text = str(value).strip()
            if not text:
                return 0.0
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                return float(text)
            except ValueError:
                return float(datetime.fromisoformat(text).timestamp())
        except Exception:
            return 0.0

    def _delete_points(
        self,
        point_ids: List[Any],
        progress_callback=None,
        progress_prefix: str = "Deleted",
        batch_size: int = 100
    ) -> int:
        """Delete points in batches with consistent error handling."""
        deleted = 0
        total = len(point_ids)
        if not total:
            return 0
        for i in range(0, total, batch_size):
            batch = point_ids[i:i+batch_size]
            try:
                if PointIdsList:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=PointIdsList(points=batch)
                    )
                else:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=batch
                    )
                deleted += len(batch)
                if progress_callback:
                    pct = (deleted / total * 100) if total > 0 else 0
                    progress_callback(f"  {progress_prefix} {deleted}/{total} ({pct:.1f}%)\n")
            except Exception as e:
                if progress_callback:
                    progress_callback(f"  Error deleting batch: {e}\n")
        return deleted

    def _upsert_points(
        self,
        points: List[Dict[str, Any]],
        progress_callback=None,
        progress_prefix: str = "Upserted",
        batch_size: int = 100
    ) -> int:
        """Upsert points in batches with consistent error handling."""
        upserted = 0
        total = len(points)
        if not total:
            return 0
        for i in range(0, total, batch_size):
            batch = points[i:i+batch_size]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                upserted += len(batch)
                if progress_callback:
                    pct = (upserted / total * 100) if total > 0 else 0
                    progress_callback(f"  {progress_prefix} {upserted}/{total} ({pct:.1f}%)\n")
            except Exception as e:
                if progress_callback:
                    progress_callback(f"  Error updating batch: {e}\n")
        return upserted

    def _build_optimizer_config(
        self,
        use_optimal: bool,
        indexing_threshold: Optional[int],
        deleted_threshold: Optional[float]
    ):
        """Build optimizer config payload for Qdrant update_collection."""
        if use_optimal:
            if indexing_threshold is None:
                indexing_threshold = 0
            if deleted_threshold is None:
                deleted_threshold = 0.1

        config_values: Dict[str, Any] = {}
        if indexing_threshold is not None:
            config_values["indexing_threshold"] = indexing_threshold
        if deleted_threshold is not None:
            config_values["deleted_threshold"] = deleted_threshold

        if not config_values:
            return None
        if OptimizersConfigDiff is not None:
            return OptimizersConfigDiff(**config_values)
        return config_values

    def _to_numpy_vector(self, v):
        """Convert vector to numpy array (handles dict/list/array)"""
        if not NUMPY_AVAILABLE:
            return None
        if isinstance(v, dict):
            key = sorted(v.keys())[0]
            v = v[key]
        return np.array(v, dtype=np.float32)

    def _run_dbscan(
        self,
        X,
        *,
        eps: float,
        min_samples: int,
        metric: str = "cosine",
        cluster_backend: str = "auto",
        progress_callback=None,
    ):
        """Run DBSCAN with GPU preference/fallback semantics."""
        backend = (cluster_backend or "auto").lower()
        if backend not in {"auto", "gpu", "cpu"}:
            raise ValueError(f"Unknown cluster backend: {cluster_backend}")

        def _log(message: str) -> None:
            if progress_callback:
                progress_callback(f"{message}\n")

        if backend in {"auto", "gpu"} and GPU_CLUSTERING_AVAILABLE and CuMLDBSCAN is not None:
            try:
                _log("Running DBSCAN clustering on GPU (cuML)...")
                gpu_input = cp.asarray(X) if cp is not None else X
                clusterer = CuMLDBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                labels = clusterer.fit_predict(gpu_input)
                if cp is not None and hasattr(labels, "get"):
                    labels = labels.get()
                return np.asarray(labels), "gpu"
            except Exception as exc:
                _log(f"GPU DBSCAN unavailable, falling back to CPU: {exc}")
        elif backend == "gpu":
            _log("GPU DBSCAN requested but cuML/CUDA dependencies are unavailable; falling back to CPU.")

        if not SKLEARN_AVAILABLE or SklearnDBSCAN is None:
            raise ImportError("sklearn not available for CPU DBSCAN fallback")

        _log("Running DBSCAN clustering on CPU (scikit-learn)...")
        clusterer = SklearnDBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
        labels = clusterer.fit_predict(X)
        return np.asarray(labels), "cpu"

    def _extract_text(self, payload: Dict) -> Optional[str]:
        """
        Extract text content from a point's payload.

        Args:
            payload: Point payload dictionary

        Returns:
            Text content or None if not found
        """
        # Extract text for fingerprinting
        text_value = None
        for field in CANONICAL_TEXT_FIELDS:
            if field not in payload:
                continue
            candidate = payload[field]
            if isinstance(candidate, str):
                if candidate.strip():
                    text_value = candidate
                    break
            elif candidate:
                text_value = candidate
                break

        return text_value

    def _score_payload_sparsity(self, payload: Dict) -> Tuple[Optional[float], Dict]:
        """
        Compute a composite payload sparsity score [0.0=dense, 1.0=sparse].
        Returns (score, breakdown) or (None, {}) if no signals could be derived.

        Signals (weighted):
          field_density  — proportion of non-empty fields          (weight 2.0, inverted)
          text_length    — primary text length normalised to 2000   (weight 1.5, inverted)
          quality_score  — any *score/*quality/*confidence field    (weight 2.0, inverted)
          nesting        — presence of nested dict values           (weight 0.5, inverted)

        Note: these heuristics reflect general assumptions about payload richness.
        They may not suit every schema — fork and adjust the signals and weights
        to match your own data model.
        """
        if not payload:
            return None, {}

        breakdown: Dict[str, float] = {}
        total_w = 0.0
        weighted = 0.0

        def _add(name: str, value: Any, weight: float, invert: bool = False) -> None:
            nonlocal total_w, weighted
            if value is None:
                return
            try:
                v = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return
            if invert:
                v = 1.0 - v
            breakdown[name] = round(v, 3)
            weighted += v * weight
            total_w += weight

        all_values = list(payload.values())

        # Signal 1: field density — how many fields are non-empty
        if all_values:
            non_empty = sum(
                1 for v in all_values
                if v is not None and v != "" and v != [] and v != {}
            )
            _add("field_density", non_empty / len(all_values), 2.0, invert=True)

        # Signal 2: text length — longer text implies denser content
        text = self._extract_text(payload)
        if text:
            _add("text_length", min(len(text) / 2000.0, 1.0), 1.5, invert=True)

        # Signal 3: quality/score/confidence — check common field naming patterns
        score_keywords = ("score", "quality", "confidence", "relevance", "rank")
        for key, val in payload.items():
            if any(kw in key.lower() for kw in score_keywords):
                try:
                    v = float(val)
                    if 0.0 <= v <= 1.0:
                        _add("quality_score", v, 2.0, invert=True)
                        break
                except (TypeError, ValueError):
                    continue

        # Signal 4: nesting — nested dicts suggest richer structured payloads
        has_nesting = any(isinstance(v, dict) for v in all_values)
        _add("nesting", 1.0 if has_nesting else 0.0, 0.5, invert=True)

        if total_w == 0.0:
            return None, {}
        return round(weighted / total_w, 4), breakdown

    def _compute_text_fingerprint(self, text: str) -> Optional[str]:
        """
        Compute a SHA256 fingerprint of normalized text content.

        Args:
            text: Text content to fingerprint

        Returns:
            SHA256 hash hex string, or None if computation fails
        """
        if not text:
            return None
        try:
            canon = re.sub(r"\s+", " ", str(text).strip()).lower()
            return hashlib.sha256(canon.encode("utf-8")).hexdigest()
        except Exception:
            return None

    def _compute_vector_health(
        self,
        vectors: list,
        sample_size: int = 1500,
        k_neighbors: int = 20,
        dup_threshold: float = 0.95,
        star_hub_threshold: float = 0.25,
        progress_callback=None
    ) -> Dict:
        """Compute advanced vector health metrics (hubness, duplicates, anisotropy)"""
        if not NUMPY_AVAILABLE:
            return {
                "error": "numpy/sklearn not available",
                "hubness": None,
                "duplicates": None,
                "anisotropy": None
            }
        
        if not vectors:
            return {"error": "no vectors", "hubness": None, "duplicates": None, "anisotropy": None}
        
        if progress_callback:
            progress_callback(f"  Computing vector health metrics (sample_size={len(vectors)})...\n")
        
        # Sample if needed
        if len(vectors) > sample_size:
            rng = random.Random(42)
            vectors = rng.sample(vectors, sample_size)
        
        X = np.stack(vectors)
        n, d = X.shape
        
        # Cosine similarity matrix
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = Xn @ Xn.T
        
        # k-NN indices (exclude self)
        k = min(k_neighbors, max(1, n - 1))
        np.fill_diagonal(S, -np.inf)
        knn_idx = np.argpartition(-S, kth=k-1, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        sorted_order = np.argsort(-S[rows, knn_idx], axis=1)
        knn = knn_idx[rows, sorted_order]
        
        # Hubness: how often each point appears in others' neighbor lists
        ref_counts = Counter(knn.flatten().tolist())
        max_refs = max(ref_counts.values()) if ref_counts else 0
        max_ref_share = max_refs / float(n * k) if n * k > 0 else 0
        star_like = max_ref_share >= star_hub_threshold
        
        # Duplicate density (upper triangle)
        upper = np.triu_indices(n, k=1)
        sims = S[upper]
        dup_pairs = int((sims >= dup_threshold).sum())
        dup_rate = dup_pairs / max(len(sims), 1)
        
        # PCA anisotropy
        n_components = min(20, d, n)
        if n_components >= 1:
            pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
            pca.fit(X)
            evr = pca.explained_variance_ratio_
            evr_cum = np.cumsum(evr)
            pc1 = float(evr[0]) if len(evr) >= 1 else 0.0
            pc2 = float(evr[1]) if len(evr) >= 2 else 0.0
            pc5 = float(evr_cum[min(4, len(evr_cum)-1)]) if len(evr_cum) >= 1 else 0.0
            pc10 = float(evr_cum[min(9, len(evr_cum)-1)]) if len(evr_cum) >= 1 else 0.0
        else:
            pc1 = pc2 = pc5 = pc10 = 0.0
        
        return {
            "hubness": {
                "max_refs": int(max_refs),
                "max_ref_share": round(max_ref_share, 4),
                "star_like": star_like,
                "star_threshold": star_hub_threshold
            },
            "duplicates": {
                "pair_rate": round(dup_rate, 4),
                "threshold": dup_threshold,
                "high_duplicates": dup_rate >= 0.02
            },
            "anisotropy": {
                "pc1_var": round(pc1, 4),
                "pc2_var": round(pc2, 4),
                "cum_pc5_var": round(pc5, 4),
                "cum_pc10_var": round(pc10, 4),
                "high_anisotropy": pc1 >= 0.20 or pc5 >= 0.60
            },
            "sample_size": n,
            "vector_dim": d
        }

    def repair_vectors(
        self,
        normalize: bool = True,
        remove_invalid: bool = True,
        dry_run: bool = False,
        progress_callback=None
    ) -> Dict:
        """
        Repair problematic vectors (normalize, remove NaN/Inf/zero vectors)
        
        Args:
            normalize: L2-normalize all vectors
            remove_invalid: Remove vectors with NaN, Inf, or zero norm
            dry_run: Preview changes without applying
            progress_callback: Progress logging function
        """
        if not self.collection_name:
            if progress_callback:
                progress_callback("Error: No collection set\n")
            return {"error": "No collection set"}
        
        if not NUMPY_AVAILABLE:
            if progress_callback:
                progress_callback("Error: numpy required for vector repair\n")
            return {"error": "numpy not available"}
        
        # Get total point count for progress tracking
        info = self.client.get_collection(self.collection_name)
        total_points = info.points_count

        if progress_callback:
            progress_callback(f"Scanning {total_points} vectors for issues...\n")

        stats = {
            "scanned": 0,
            "invalid_vectors": [],
            "zero_norm": [],
            "needs_normalization": [],
            "repaired": 0,
            "removed": 0
        }

        offset = None
        batch_updates = []

        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )

            if not results:
                break

            for point in results:
                stats["scanned"] += 1
                
                if not hasattr(point, 'vector') or not point.vector:
                    stats["invalid_vectors"].append(point.id)
                    continue
                
                vec = self._to_numpy_vector(point.vector)
                if vec is None:
                    stats["invalid_vectors"].append(point.id)
                    continue
                
                # Check for NaN or Inf
                if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                    stats["invalid_vectors"].append(point.id)
                    continue
                
                # Check for zero norm
                norm = np.linalg.norm(vec)
                if norm < 1e-10:
                    stats["zero_norm"].append(point.id)
                    continue
                
                # Check if normalization needed
                if normalize and abs(norm - 1.0) > 1e-6:
                    stats["needs_normalization"].append(point.id)
                    normalized_vec = vec / norm
                    
                    # Prepare update
                    if isinstance(point.vector, dict):
                        # Multi-vector: update first vector
                        key = sorted(point.vector.keys())[0]
                        updated_vector = dict(point.vector)
                        updated_vector[key] = normalized_vec.tolist()
                    else:
                        updated_vector = normalized_vec.tolist()
                    
                    batch_updates.append({
                        "id": point.id,
                        "vector": updated_vector,
                        "payload": point.payload
                    })
            
            if progress_callback and stats["scanned"] % 500 == 0:
                pct = (stats["scanned"] / total_points * 100) if total_points > 0 else 0
                progress_callback(f"  Progress: {stats['scanned']}/{total_points} ({pct:.1f}%) - Found {len(stats['invalid_vectors'])} invalid, {len(stats['zero_norm'])} zero-norm, {len(stats['needs_normalization'])} needing normalization\n")
            
            if offset is None:
                break
        
        # Summary
        if progress_callback:
            progress_callback(f"\nScan complete: {stats['scanned']} vectors\n")
            progress_callback(f"  Invalid (NaN/Inf): {len(stats['invalid_vectors'])}\n")
            progress_callback(f"  Zero norm: {len(stats['zero_norm'])}\n")
            progress_callback(f"  Need normalization: {len(stats['needs_normalization'])}\n")
        
        if dry_run:
            return stats
        
        # Apply repairs
        if remove_invalid and (stats["invalid_vectors"] or stats["zero_norm"]):
            to_delete = stats["invalid_vectors"] + stats["zero_norm"]
            if progress_callback:
                progress_callback(f"\nRemoving {len(to_delete)} invalid vectors...\n")
            stats["removed"] = self._delete_points(
                to_delete,
                progress_callback=progress_callback,
                progress_prefix="Removed"
            )
        
        if normalize and batch_updates:
            if progress_callback:
                progress_callback(f"\nNormalizing {len(batch_updates)} vectors...\n")
            stats["repaired"] = self._upsert_points(
                batch_updates,
                progress_callback=progress_callback,
                progress_prefix="Normalized"
            )
        
        if progress_callback:
            progress_callback(f"\nRepair complete:\n")
            progress_callback(f"  Removed: {stats['removed']}\n")
            progress_callback(f"  Normalized: {stats['repaired']}\n")
        
        return stats

    def semantic_deduplication(
        self,
        similarity_threshold: float = 0.98,
        keep_strategy: str = "newest",
        use_clustering: bool = True,
        cluster_backend: str = "auto",
        min_cluster_size: int = 2,
        dry_run: bool = False,
        progress_callback=None
    ) -> Dict:
        """
        Advanced semantic deduplication using clustering
        
        Args:
            similarity_threshold: Cosine similarity threshold for duplicates (0.95-0.99)
            keep_strategy: Which duplicate to keep ('newest', 'oldest', 'most_complete')
            use_clustering: Use DBSCAN clustering vs pairwise comparison
            cluster_backend: DBSCAN backend ('auto', 'gpu', 'cpu')
            min_cluster_size: Minimum cluster size to consider as duplicates
            dry_run: Preview without deleting
            progress_callback: Progress logging function
        """
        if not self.collection_name:
            if progress_callback:
                progress_callback("Error: No collection set\n")
            return {"error": "No collection set"}
        
        if not NUMPY_AVAILABLE:
            if progress_callback:
                progress_callback("Error: numpy/sklearn required\n")
            return {"error": "numpy not available"}
        
        if not SKLEARN_AVAILABLE:
            use_clustering = False
            if progress_callback:
                progress_callback("Warning: sklearn not available, using pairwise comparison\n")
        
        if progress_callback:
            progress_callback(f"Semantic deduplication (threshold={similarity_threshold}, strategy={keep_strategy})\n")

        # Get total point count for progress tracking
        info = self.client.get_collection(self.collection_name)
        total_points = info.points_count

        # Collect all vectors
        if progress_callback:
            progress_callback(f"Collecting {total_points} vectors...\n")

        points_data = []
        offset = None
        
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            if not results:
                break
            
            for point in results:
                if not hasattr(point, 'vector') or not point.vector:
                    continue
                
                vec = self._to_numpy_vector(point.vector)
                if vec is None:
                    continue
                
                payload = point.payload or {}
                
                # Extract timestamp for keep_strategy
                timestamp = 0
                for field in ("created_at", "timestamp", "updated_at"):
                    if field in payload:
                        timestamp = self._parse_timestamp(payload[field])
                        if timestamp:
                            break
                
                completeness = sum(1 for v in payload.values() if v not in (None, "", [], {}, ()))

                points_data.append({
                    "id": point.id,
                    "vector": vec,
                    "timestamp": timestamp,
                    "completeness": completeness
                })

            if progress_callback and len(points_data) % 500 == 0:
                pct = (len(points_data) / total_points * 100) if total_points > 0 else 0
                progress_callback(f"  Progress: {len(points_data)}/{total_points} ({pct:.1f}%)\n")
            
            if offset is None:
                break
        
        if not points_data:
            if progress_callback:
                progress_callback("No vectors found\n")
            return {"duplicate_groups": [], "to_delete": [], "kept": []}
        
        if progress_callback:
            progress_callback(f"Collected {len(points_data)} vectors\n")
        
        # Stack vectors
        X = np.stack([p["vector"] for p in points_data])
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        
        duplicate_groups = []
        
        if use_clustering:
            # DBSCAN clustering in cosine space
            # Convert similarity to distance: distance = 1 - similarity
            eps = 1.0 - similarity_threshold

            labels, backend_used = self._run_dbscan(
                X_norm,
                eps=eps,
                min_samples=min_cluster_size,
                metric="cosine",
                cluster_backend=cluster_backend,
                progress_callback=progress_callback,
            )
            
            # Group by cluster label (ignore noise: label=-1)
            clusters = defaultdict(list)
            for idx, label in enumerate(labels):
                if label != -1:
                    clusters[label].append(idx)
            
            # Each cluster is a duplicate group
            for cluster_id, indices in clusters.items():
                if len(indices) >= min_cluster_size:
                    group = [points_data[idx] for idx in indices]
                    duplicate_groups.append(group)

            if progress_callback:
                progress_callback(f"Found {len(duplicate_groups)} duplicate clusters using {backend_used.upper()} backend\n")
        
        else:
            # Pairwise comparison (slower but more precise)
            if progress_callback:
                progress_callback(f"Computing pairwise similarities...\n")
            
            S = X_norm @ X_norm.T
            
            # Find connected components of high-similarity pairs
            visited = set()
            for i in range(len(points_data)):
                if i in visited:
                    continue
                
                # Find all points similar to i
                similar = [i]
                for j in range(i + 1, len(points_data)):
                    if S[i, j] >= similarity_threshold:
                        similar.append(j)
                
                if len(similar) >= min_cluster_size:
                    group = [points_data[idx] for idx in similar]
                    duplicate_groups.append(group)
                    visited.update(similar)
            
            if progress_callback:
                progress_callback(f"Found {len(duplicate_groups)} duplicate groups\n")
        
        # Decide which to keep in each group
        to_delete = []
        kept = []
        
        for group in duplicate_groups:
            if keep_strategy == "newest":
                sorted_group = sorted(group, key=lambda p: p["timestamp"], reverse=True)
            elif keep_strategy == "oldest":
                sorted_group = sorted(group, key=lambda p: p["timestamp"], reverse=False)
            else:  # most_complete
                sorted_group = sorted(group, key=lambda p: p["completeness"], reverse=True)
            
            kept.append(sorted_group[0]["id"])
            to_delete.extend([p["id"] for p in sorted_group[1:]])
        
        stats = {
            "duplicate_groups": len(duplicate_groups),
            "total_duplicates": len(to_delete),
            "to_delete": to_delete,
            "kept": kept
        }
        
        if progress_callback:
            progress_callback(f"\nFound {stats['duplicate_groups']} groups, {stats['total_duplicates']} duplicates to remove\n")
        
        if dry_run:
            return stats
        
        # Delete duplicates
        if to_delete:
            if progress_callback:
                progress_callback(f"Deleting {len(to_delete)} duplicates...\n")
            stats["deleted"] = self._delete_points(
                to_delete,
                progress_callback=progress_callback,
                progress_prefix="Deleted"
            )
            if progress_callback:
                progress_callback(f"Deleted {stats['deleted']} duplicates\n")
        
        return stats

    def detect_outliers(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.01,
        cluster_backend: str = "auto",
        eps: float = 0.3,
        min_samples: int = 5,
        remove: bool = False,
        dry_run: bool = False,
        progress_callback=None
    ) -> Dict:
        """
        Detect and optionally remove outlier vectors

        Args:
            method: Detection method ('isolation_forest', 'dbscan', 'statistical')
            contamination: Expected proportion of outliers (0.001-0.1)
            cluster_backend: DBSCAN backend ('auto', 'gpu', 'cpu')
            eps: DBSCAN neighbourhood radius (cosine distance)
            min_samples: DBSCAN minimum neighbours to form a core point
            remove: Delete detected outliers
            dry_run: Preview without deleting
            progress_callback: Progress logging function
        """
        if not self.collection_name:
            if progress_callback:
                progress_callback("Error: No collection set\n")
            return {"error": "No collection set"}
        
        if not NUMPY_AVAILABLE:
            if progress_callback:
                progress_callback("Error: numpy/sklearn required\n")
            return {"error": "numpy not available"}
        
        if method == "isolation_forest" and (not SKLEARN_AVAILABLE or IsolationForest is None):
            if progress_callback:
                progress_callback("Error: sklearn required\n")
            return {"error": "sklearn not available"}
        
        if progress_callback:
            progress_callback(f"Detecting outliers (method={method}, contamination={contamination})...\n")

        # Get total point count for progress tracking
        info = self.client.get_collection(self.collection_name)
        total_points = info.points_count

        # Collect vectors
        if progress_callback:
            progress_callback(f"Collecting {total_points} vectors...\n")

        points_data = []
        offset = None
        
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=True
            )
            
            if not results:
                break
            
            for point in results:
                if not hasattr(point, 'vector') or not point.vector:
                    continue
                
                vec = self._to_numpy_vector(point.vector)
                if vec is None:
                    continue

                points_data.append({"id": point.id, "vector": vec})

            if progress_callback and len(points_data) % 500 == 0:
                pct = (len(points_data) / total_points * 100) if total_points > 0 else 0
                progress_callback(f"  Progress: {len(points_data)}/{total_points} ({pct:.1f}%)\n")
            
            if offset is None:
                break
        
        if not points_data:
            return {"outliers": [], "removed": 0}
        
        X = np.stack([p["vector"] for p in points_data])
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        if progress_callback:
            progress_callback(f"Running {method} on {len(points_data)} vectors...\n")

        outlier_mask = None

        if method == "isolation_forest":
            clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            predictions = clf.fit_predict(X_norm)
            outlier_mask = predictions == -1

        elif method == "dbscan":
            # DBSCAN: points labeled as -1 are noise/outliers
            labels, backend_used = self._run_dbscan(
                X_norm,
                eps=eps,
                min_samples=min_samples,
                metric="cosine",
                cluster_backend=cluster_backend,
                progress_callback=progress_callback,
            )
            outlier_mask = labels == -1
            if progress_callback:
                progress_callback(f"DBSCAN outlier detection used {backend_used.upper()} backend\n")
        
        elif method == "statistical":
            # Statistical: z-score > 3 in any dimension
            z_scores = np.abs((X_norm - np.mean(X_norm, axis=0)) / (np.std(X_norm, axis=0) + 1e-10))
            outlier_mask = np.any(z_scores > 3, axis=1)
        
        else:
            if progress_callback:
                progress_callback(f"Unknown method: {method}\n")
            return {"error": f"Unknown method: {method}"}
        
        outlier_ids = [points_data[i]["id"] for i in range(len(points_data)) if outlier_mask[i]]
        
        stats = {
            "total_vectors": len(points_data),
            "outliers_detected": len(outlier_ids),
            "outlier_rate": len(outlier_ids) / len(points_data) if points_data else 0,
            "outlier_ids": outlier_ids,
            "removed": 0
        }
        
        if progress_callback:
            progress_callback(f"Detected {len(outlier_ids)} outliers ({stats['outlier_rate']:.2%})\n")
        
        if dry_run or not remove:
            return stats
        
        # Remove outliers
        if outlier_ids:
            if progress_callback:
                progress_callback(f"Removing {len(outlier_ids)} outliers...\n")
            stats["removed"] = self._delete_points(
                outlier_ids,
                progress_callback=progress_callback,
                progress_prefix="Removed"
            )
            if progress_callback:
                progress_callback(f"Removed {stats['removed']} outliers\n")

        return stats

    def scan_sparse(
        self,
        percentile: float = 0.1,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        vector_weight: float = 0.55,
        action: str = "report",
        dry_run: bool = True,
        progress_callback=None,
    ) -> Dict:
        """
        Full-pass semantic sparsity scan combining IsolationForest vector anomaly scores
        with payload metadata signals.

        Single scroll collects both vectors and payload.  After the scroll, IsolationForest
        is fit on all vectors and decision_function() is used to produce a continuous
        anomaly score (more negative = more anomalous) which is flipped and min-max
        normalised to [0, 1].  That vector score is combined with the metadata score:

            final = vector_weight * vec_score + (1 - vector_weight) * meta_score

        Graceful fallbacks:
          - No vector  → use metadata score at full weight
          - No metadata signal → use vector score at full weight
          - Neither    → point is skipped

        Args:
            percentile:     Flag worst N% (0.1 = bottom 10%). Used when threshold/top_n absent.
            threshold:      Flag every point with final_score >= threshold.
            top_n:          Flag the N sparsest points.
            vector_weight:  Weight for vector anomaly score (0.0–1.0); remainder goes to metadata.
            action:         "report" | "tag" | "delete"
                            "tag" writes _sparsity_score to each candidate's payload.
                            "delete" removes candidates (respects dry_run).
            dry_run:        Preview without writing changes.
            progress_callback: Progress logging function.
        """
        if not self.collection_name:
            if progress_callback:
                progress_callback("Error: No collection set\n")
            return {"error": "No collection set"}

        if not NUMPY_AVAILABLE:
            if progress_callback:
                progress_callback("Warning: numpy/sklearn unavailable — falling back to metadata-only scoring\n")

        try:
            from sklearn.ensemble import IsolationForest as _IF
        except ImportError:
            _IF = None
            if progress_callback:
                progress_callback("Warning: sklearn unavailable — falling back to metadata-only scoring\n")

        meta_weight = 1.0 - vector_weight

        info = self.client.get_collection(self.collection_name)
        total_points = info.points_count

        if progress_callback:
            progress_callback(f"Collecting {total_points} points (vectors + payload)...\n")
            if NUMPY_AVAILABLE and _IF is not None:
                est_mb = total_points * 1536 * 4 / 1_048_576
                progress_callback(f"  Estimated vector memory: ~{est_mb:.0f} MB\n")

        points_data: List[Dict] = []
        no_signal_count = 0
        offset = None
        scanned = 0

        use_vectors = NUMPY_AVAILABLE and _IF is not None

        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=use_vectors,
            )

            if not results:
                break

            for point in results:
                scanned += 1
                payload = point.payload or {}
                meta_score, breakdown = self._score_payload_sparsity(payload)

                vec = None
                if use_vectors and hasattr(point, "vector") and point.vector:
                    vec = self._to_numpy_vector(point.vector)

                if meta_score is None and vec is None:
                    no_signal_count += 1
                    continue

                meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                title = payload.get("title") or meta.get("title") or ""
                url = payload.get("url") or meta.get("url") or ""
                content_type = (
                    payload.get("content_type") or meta.get("content_type")
                    or payload.get("format") or ""
                )

                points_data.append({
                    "id": point.id,
                    "vec": vec,
                    "meta_score": meta_score,
                    "breakdown": breakdown,
                    "title": title[:80] if title else (url[:80] if url else ""),
                    "content_type": content_type,
                    "vec_score": None,
                    "final_score": None,
                })

            if progress_callback and scanned % 1000 == 0:
                pct = (scanned / total_points * 100) if total_points > 0 else 0
                progress_callback(f"  Collected {scanned}/{total_points} ({pct:.1f}%)\n")

            if offset is None:
                break

        if progress_callback:
            progress_callback(f"Collected {len(points_data)} scorable points ({no_signal_count} skipped — no signal)\n")

        # ── Vector scoring ─────────────────────────────────────────────────────
        vec_indices = [i for i, p in enumerate(points_data) if p["vec"] is not None]

        if vec_indices and NUMPY_AVAILABLE and _IF is not None:
            if progress_callback:
                progress_callback(f"Fitting IsolationForest on {len(vec_indices)} vectors...\n")

            X = np.stack([points_data[i]["vec"] for i in vec_indices])

            clf = _IF(contamination="auto", n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X)

            if progress_callback:
                progress_callback("  Computing anomaly scores...\n")

            raw = clf.decision_function(X)   # higher = more normal
            anomaly = -raw                    # flip: higher = more anomalous
            lo, hi = anomaly.min(), anomaly.max()
            if hi > lo:
                vec_scores_norm = (anomaly - lo) / (hi - lo)
            else:
                vec_scores_norm = np.zeros(len(anomaly))

            for j, idx in enumerate(vec_indices):
                points_data[idx]["vec_score"] = float(vec_scores_norm[j])

            # Free RAM — vectors no longer needed
            for p in points_data:
                p["vec"] = None

        # ── Combine scores ─────────────────────────────────────────────────────
        for p in points_data:
            vs = p["vec_score"]
            ms = p["meta_score"]
            if vs is not None and ms is not None:
                p["final_score"] = round(vector_weight * vs + meta_weight * ms, 4)
                p["breakdown"]["vec_score"] = round(vs, 3)
                p["breakdown"]["meta_score"] = round(ms, 3)
            elif vs is not None:
                p["final_score"] = round(vs, 4)
                p["breakdown"]["vec_score"] = round(vs, 3)
            else:
                p["final_score"] = round(ms, 4)
                p["breakdown"]["meta_score"] = round(ms, 3)

        # Filter points that couldn't be scored at all (shouldn't happen but be safe)
        scorable = [p for p in points_data if p["final_score"] is not None]
        scorable.sort(key=lambda x: x["final_score"], reverse=True)

        if progress_callback:
            progress_callback(f"Scored {len(scorable)} points\n")

        # ── Apply cutoff ───────────────────────────────────────────────────────
        if threshold is not None:
            candidates = [p for p in scorable if p["final_score"] >= threshold]
            cutoff_desc = f"score >= {threshold}"
        elif top_n is not None:
            candidates = scorable[:top_n]
            cutoff_desc = f"top {top_n}"
        else:
            cutoff = max(1, int(len(scorable) * percentile))
            candidates = scorable[:cutoff]
            cutoff_desc = f"bottom {percentile:.0%}"

        if progress_callback:
            progress_callback(f"Candidates: {len(candidates)} ({cutoff_desc})\n")

        candidate_ids = [c["id"] for c in candidates]

        stats: Dict = {
            "total_points": total_points,
            "scored": len(scorable),
            "no_signal": no_signal_count,
            "candidates": len(candidates),
            "candidate_ids": candidate_ids,
            "top_candidates": candidates[:50],
            "tagged": 0,
            "deleted": 0,
            "action": action,
            "dry_run": dry_run,
            "vector_weight": vector_weight,
            "meta_weight": meta_weight,
            "vec_scored_count": len(vec_indices),
        }

        # ── Actions ────────────────────────────────────────────────────────────
        if action == "tag" and not dry_run:
            if progress_callback:
                progress_callback(f"\nTagging {len(candidates)} points with _sparsity_score...\n")
            tagged = 0
            for cand in candidates:
                try:
                    self.client.set_payload(
                        collection_name=self.collection_name,
                        payload={"_sparsity_score": cand["final_score"]},
                        points=[cand["id"]],
                    )
                    tagged += 1
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"  Error tagging {cand['id']}: {e}\n")
            stats["tagged"] = tagged
            if progress_callback:
                progress_callback(f"Tagged {tagged} points\n")

        elif action == "delete" and not dry_run:
            if candidate_ids:
                if progress_callback:
                    progress_callback(f"\nDeleting {len(candidate_ids)} sparse points...\n")
                stats["deleted"] = self._delete_points(
                    candidate_ids,
                    progress_callback=progress_callback,
                    progress_prefix="Deleted",
                )
                if progress_callback:
                    progress_callback(f"Deleted {stats['deleted']} sparse points\n")

        return stats

    def analyze_collection(
        self,
        progress_callback=None,
        keep_per_group: int = 1,
        enable_vector_analysis: bool = True,
        vector_sample_size: int = 1500
    ) -> Dict:
        """Analyze collection for duplicates and health"""
        if not self.collection_name:
            if progress_callback:
                progress_callback("Error: No collection set\n")
            return {}
        
        info = self.client.get_collection(self.collection_name)
        total_points = info.points_count
        
        if progress_callback:
            progress_callback(f"Analyzing {self.collection_name} ({total_points} points)\n")
        
        groups = defaultdict(list)
        empty_vectors = []
        missing_text = []
        issues = []

        # Metadata statistics tracking
        metadata_fields = defaultdict(int)  # Count of points with each field
        metadata_values = defaultdict(lambda: defaultdict(int))  # Count of values for each field
        metadata_types = defaultdict(set)  # Track types for each field

        # Collect vectors for health analysis
        vector_samples = [] if enable_vector_analysis and NUMPY_AVAILABLE else None

        offset = None
        scanned = 0
        
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=enable_vector_analysis
            )
            
            if not results:
                break
            
            for point in results:
                scanned += 1
                payload = point.payload or {}

                # Collect metadata statistics
                for key, value in payload.items():
                    metadata_fields[key] += 1
                    value_type = type(value).__name__
                    metadata_types[key].add(value_type)

                    # Track value distribution (limit to reasonable values)
                    if isinstance(value, (str, int, float, bool)):
                        # Truncate long strings for counting
                        str_value = str(value)
                        if len(str_value) > 100:
                            str_value = str_value[:100] + "..."
                        metadata_values[key][str_value] += 1
                    elif isinstance(value, dict):
                        # For nested metadata, track nested keys
                        for nested_key in value.keys():
                            nested_field = f"{key}.{nested_key}"
                            metadata_fields[nested_field] += 1
                            if isinstance(value[nested_key], (str, int, float, bool)):
                                nested_str = str(value[nested_key])
                                if len(nested_str) > 100:
                                    nested_str = nested_str[:100] + "..."
                                metadata_values[nested_field][nested_str] += 1
                                metadata_types[nested_field].add(type(value[nested_key]).__name__)

                text_value = self._extract_text(payload)
                
                if not text_value:
                    missing_text.append(point.id)
                
                # Check vector
                vector_present = False
                if hasattr(point, 'vector') and point.vector:
                    vec = point.vector
                    if isinstance(vec, dict):
                        vec = list(vec.values())[0] if vec else None
                    vector_present = bool(vec)
                    
                    # Collect vector for health analysis
                    if vector_present and vector_samples is not None and len(vector_samples) < vector_sample_size:
                        np_vec = self._to_numpy_vector(point.vector)
                        if np_vec is not None:
                            vector_samples.append(np_vec)
                
                if not vector_present:
                    empty_vectors.append(point.id)
                    if text_value:  # Only skip grouping if no text
                        continue
                
                if not text_value:
                    continue

                # Compute fingerprint
                fingerprint = self._compute_text_fingerprint(text_value)
                if fingerprint:
                    # Completeness score
                    completeness = sum(1 for v in payload.values() if v not in (None, "", [], {}, ()))
                    
                    # Timestamp
                    created_ts = 0
                    for tfield in ("created_at", "timestamp", "updated_at"):
                        if tfield in payload and payload[tfield]:
                            created_ts = self._parse_timestamp(payload[tfield])
                            if created_ts:
                                break
                    
                    groups[fingerprint].append((point.id, vector_present, completeness, created_ts))
            
            if progress_callback and scanned % 500 == 0:
                progress_callback(f"  Scanned {scanned}/{total_points}\n")
            
            if offset is None:
                break
        
        # Calculate duplicates
        duplicate_groups = {k: v for k, v in groups.items() if len(v) > keep_per_group}
        duplicate_count = sum(len(v) - keep_per_group for v in duplicate_groups.values())
        
        # Compute vector health metrics
        vector_health = None
        if vector_samples and len(vector_samples) > 0:
            vector_health = self._compute_vector_health(
                vector_samples,
                sample_size=vector_sample_size,
                progress_callback=progress_callback
            )
        
        # Health score
        health_score = 100.0
        if empty_vectors:
            health_score -= min(30, len(empty_vectors) / total_points * 100)
        if missing_text:
            health_score -= min(30, len(missing_text) / total_points * 100)
        if duplicate_count:
            health_score -= min(40, duplicate_count / total_points * 100)
        
        # Adjust health based on vector analysis
        if vector_health and not vector_health.get("error"):
            if vector_health["hubness"]["star_like"]:
                health_score -= 15
            if vector_health["duplicates"]["high_duplicates"]:
                health_score -= 10
            if vector_health["anisotropy"]["high_anisotropy"]:
                health_score -= 10
        
        # Build issues list
        if empty_vectors:
            issues.append(f"{len(empty_vectors)} points with empty vectors")
        if missing_text:
            issues.append(f"{len(missing_text)} points with missing text")
        if duplicate_count:
            issues.append(f"{duplicate_count} duplicate points across {len(duplicate_groups)} groups")
        
        if vector_health and not vector_health.get("error"):
            if vector_health["hubness"]["star_like"]:
                share = vector_health["hubness"]["max_ref_share"]
                issues.append(f"Star-like topology detected ({share:.1%} of neighbors point to single hub)")
            if vector_health["duplicates"]["high_duplicates"]:
                rate = vector_health["duplicates"]["pair_rate"]
                issues.append(f"High vector similarity: {rate:.1%} of pairs above threshold")
            if vector_health["anisotropy"]["high_anisotropy"]:
                pc1 = vector_health["anisotropy"]["pc1_var"]
                issues.append(f"Anisotropic vectors: PC1 explains {pc1:.1%} of variance")

        # Print detailed metadata statistics
        if progress_callback:
            progress_callback("\n" + "=" * 70 + "\n")
            progress_callback("METADATA STATISTICS\n")
            progress_callback("=" * 70 + "\n\n")

            # Sort fields by frequency (most common first)
            sorted_fields = sorted(metadata_fields.items(), key=lambda x: x[1], reverse=True)

            progress_callback(f"Total metadata fields found: {len(sorted_fields)}\n\n")

            for field, count in sorted_fields:
                pct = (count / total_points * 100) if total_points > 0 else 0
                types_str = ", ".join(sorted(metadata_types[field]))
                progress_callback(f"Field: {field}\n")
                progress_callback(f"  Present in: {count}/{total_points} points ({pct:.1f}%)\n")
                progress_callback(f"  Types: {types_str}\n")

                # Show value distribution for fields with limited unique values
                if field in metadata_values:
                    unique_values = len(metadata_values[field])
                    if unique_values <= 20:  # Only show distribution for fields with <= 20 unique values
                        progress_callback(f"  Unique values: {unique_values}\n")
                        progress_callback(f"  Value distribution:\n")

                        # Sort by count (most common first)
                        sorted_values = sorted(metadata_values[field].items(), key=lambda x: x[1], reverse=True)
                        for value, value_count in sorted_values[:10]:  # Show top 10 values
                            value_pct = (value_count / count * 100) if count > 0 else 0
                            progress_callback(f"    - {value}: {value_count} ({value_pct:.1f}%)\n")

                        if len(sorted_values) > 10:
                            progress_callback(f"    ... and {len(sorted_values) - 10} more values\n")
                    else:
                        progress_callback(f"  Unique values: {unique_values} (too many to display)\n")

                progress_callback("\n")

            progress_callback("=" * 70 + "\n\n")

        return {
            "total_points": total_points,
            "duplicate_count": duplicate_count,
            "duplicate_groups": duplicate_groups,
            "health_score": round(max(0, health_score), 1),
            "issues": issues,
            "empty_vectors": empty_vectors,
            "missing_text": missing_text,
            "segments_count": info.segments_count,
            "points_count": info.points_count,
            "status": info.status,
            "vector_health": vector_health,
            "metadata_statistics": {
                "fields": dict(metadata_fields),
                "values": {k: dict(v) for k, v in metadata_values.items()},
                "types": {k: list(v) for k, v in metadata_types.items()}
            }
        }

    def remove_duplicates(
        self,
        dry_run: bool = False,
        progress_callback=None,
        keep_per_group: int = 1
    ) -> int:
        """Remove duplicate points"""
        if not self.collection_name:
            if progress_callback:
                progress_callback("Error: No collection set\n")
            return 0

        # Get total point count for progress tracking
        info = self.client.get_collection(self.collection_name)
        total_points = info.points_count

        if progress_callback:
            progress_callback(f"Scanning {total_points} points for duplicates (hash, keep={keep_per_group})...\n")

        groups = defaultdict(list)
        offset = None
        scanned = 0
        
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            if not results:
                break
            
            for point in results:
                scanned += 1
                payload = point.payload or {}

                text_value = self._extract_text(payload)
                fingerprint = self._compute_text_fingerprint(text_value) if text_value else None

                vector_present = False
                if hasattr(point, 'vector') and point.vector:
                    vec = point.vector
                    if isinstance(vec, dict):
                        vec = list(vec.values())[0] if vec else None
                    vector_present = bool(vec)

                completeness = sum(1 for v in payload.values() if v not in (None, "", [], {}, ()))

                created_ts = 0
                for tfield in ("created_at", "timestamp", "updated_at"):
                    if tfield in payload and payload[tfield]:
                        created_ts = self._parse_timestamp(payload[tfield])
                        if created_ts:
                            break

                if fingerprint:
                    groups[fingerprint].append((point.id, vector_present, completeness, created_ts))

            if progress_callback and scanned % 500 == 0:
                pct = (scanned / total_points * 100) if total_points > 0 else 0
                progress_callback(f"  Progress: {scanned}/{total_points} ({pct:.1f}%) - Found {len(groups)} unique groups\n")

            if offset is None:
                break
        
        # Select duplicates to delete
        to_delete = []
        for group_key, metas in groups.items():
            if len(metas) <= keep_per_group:
                continue
            
            sorted_metas = sorted(
                metas,
                key=lambda m: (1 if m[1] else 0, m[2], m[3]),
                reverse=True
            )
            
            to_delete.extend([m[0] for m in sorted_metas[keep_per_group:]])
        
        if not to_delete:
            if progress_callback:
                progress_callback("No duplicates found\n")
            return 0
        
        if progress_callback:
            progress_callback(f"Found {len(to_delete)} duplicates\n")
        
        if dry_run:
            return len(to_delete)
        
        # Delete in batches
        deleted = self._delete_points(
            to_delete,
            progress_callback=progress_callback,
            progress_prefix="Deleted"
        )
        
        if progress_callback:
            progress_callback(f"Removed {deleted} duplicates\n")
        
        return deleted

    def optimize_collection(
        self,
        progress_callback=None,
        *,
        use_optimal: bool = True,
        indexing_threshold: Optional[int] = None,
        deleted_threshold: Optional[float] = None
    ):
        """Optimize collection"""
        if progress_callback:
            progress_callback(f"Optimizing {self.collection_name}...\n")
        optimizer_config = self._build_optimizer_config(
            use_optimal=use_optimal,
            indexing_threshold=indexing_threshold,
            deleted_threshold=deleted_threshold
        )

        try:
            if optimizer_config:
                try:
                    self.client.update_collection(
                        collection_name=self.collection_name,
                        optimizer_config=optimizer_config
                    )
                    if progress_callback:
                        progress_callback("Optimizer config updated\n")
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Optimizer config update failed: {e}\n")

            if self.has_optimize_method:
                try:
                    self.client.optimize_collection(collection_name=self.collection_name, wait=True)
                    if progress_callback:
                        progress_callback("Optimization complete\n")
                    return
                except TypeError:
                    self.client.optimize_collection(collection_name=self.collection_name)
                    if progress_callback:
                        progress_callback("Optimization triggered\n")
                    return
        except Exception as e:
            if progress_callback:
                progress_callback(f"Modern optimize failed: {e}\n")
        
        # Legacy fallback - use dict directly instead of OptimizersConfigDiff
        try:
            if optimizer_config:
                self.client.update_collection(
                    collection_name=self.collection_name,
                    optimizer_config=optimizer_config
                )
                if progress_callback:
                    progress_callback("Optimization triggered (legacy)\n")
            else:
                if progress_callback:
                    progress_callback("Optimization skipped (no optimizer config)\n")
        except Exception as e:
            if progress_callback:
                progress_callback(f"Optimization failed: {e}\n")

    def one_stop_clean(
        self,
        remove_duplicates: bool = True,
        optimize: bool = True,
        dry_run: bool = False,
        progress_callback=None,
        keep_per_group: int = 1,
        optimize_use_optimal: bool = True,
        optimize_indexing_threshold: Optional[int] = None,
        optimize_deleted_threshold: Optional[float] = None
    ) -> Dict:
        """One-stop collection cleaning"""
        if progress_callback:
            progress_callback("=" * 60 + "\n")
            progress_callback("Starting one-stop clean\n")
            progress_callback("=" * 60 + "\n")

        removed = 0
        if remove_duplicates:
            if progress_callback:
                progress_callback("\n[Step 1/2] Removing duplicates...\n")
            removed = self.remove_duplicates(
                dry_run=dry_run,
                progress_callback=progress_callback,
                keep_per_group=keep_per_group
            )

        if optimize and not dry_run:
            if progress_callback:
                progress_callback("\n[Step 2/2] Optimizing collection...\n")
            self.optimize_collection(
                progress_callback=progress_callback,
                use_optimal=optimize_use_optimal,
                indexing_threshold=optimize_indexing_threshold,
                deleted_threshold=optimize_deleted_threshold
            )

        if progress_callback:
            progress_callback("\n" + "=" * 60 + "\n")
            progress_callback(f"One-stop clean complete - Removed {removed} duplicates\n")
            progress_callback("=" * 60 + "\n")

        return {"removed": removed}

    def remove_points_by_payload_filter(
        self,
        filters: Dict[str, str],
        dry_run: bool = False,
        progress_callback=None
    ) -> Dict:
        """
        Remove points where all specified payload fields match the given values.

        Args:
            filters: Mapping of payload field name to required value. All
                     conditions must match (logical AND).
            dry_run: Preview without deleting
            progress_callback: Progress logging function

        Returns:
            Dictionary with 'total_found', 'removed', and 'filter' keys
        """
        if not self.collection_name:
            if progress_callback:
                progress_callback("Error: No collection set\n")
            return {"error": "No collection set"}

        if not filters:
            if progress_callback:
                progress_callback("Error: No filters specified\n")
            return {"error": "No filters specified"}

        from .dependencies import qdrant_models

        if qdrant_models is None:
            if progress_callback:
                progress_callback("Error: qdrant-client required\n")
            return {"error": "qdrant-client not available"}

        filter_summary = ", ".join(f"{k}={v!r}" for k, v in filters.items())
        if progress_callback:
            progress_callback(f"Scanning with filter: {filter_summary}\n")

        filter_conditions = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key=field,
                    match=qdrant_models.MatchValue(value=value)
                )
                for field, value in filters.items()
            ]
        )

        matching_ids = []
        field_value_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        offset = None

        while True:
            try:
                results, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_conditions,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error during scroll: {e}\n")
                return {"error": f"Scroll failed: {e}"}

            if not results:
                break

            for point in results:
                matching_ids.append(point.id)
                payload = point.payload or {}
                for field in filters:
                    field_value_counts[field][str(payload.get(field, "unknown"))] += 1

            if progress_callback and len(matching_ids) % 500 == 0:
                progress_callback(f"  Found {len(matching_ids)} matching points...\n")

            if offset is None:
                break

        stats: Dict[str, Any] = {
            "total_found": len(matching_ids),
            "field_counts": {k: dict(v) for k, v in field_value_counts.items()},
            "removed": 0,
            "filter": dict(filters),
        }

        if progress_callback:
            progress_callback(f"\nFound {len(matching_ids)} matching points\n")
            for field, counts in field_value_counts.items():
                progress_callback(f"\n{field} breakdown:\n")
                for val, count in counts.items():
                    progress_callback(f"  {val}: {count}\n")

        if not matching_ids:
            return stats

        if dry_run:
            if progress_callback:
                progress_callback(f"\n[DRY RUN] Would delete {len(matching_ids)} points\n")
            return stats

        if progress_callback:
            progress_callback(f"\nDeleting {len(matching_ids)} points...\n")
        stats["removed"] = self._delete_points(
            matching_ids,
            progress_callback=progress_callback,
            progress_prefix="Deleted"
        )

        if progress_callback:
            progress_callback(f"\nDeleted {stats['removed']} points\n")

        return stats
