"""Optional dependency imports for the qdrant manager tool."""
from __future__ import annotations

NUMPY_AVAILABLE = False
SKLEARN_AVAILABLE = False
GPU_CLUSTERING_AVAILABLE = False
np = None
pd = None
PCA = None
SklearnDBSCAN = None
IsolationForest = None
CuMLDBSCAN = None
cp = None

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore[assignment]
    import pandas as pd  # type: ignore[assignment]
    from sklearn.decomposition import PCA  # type: ignore[assignment]
    from sklearn.cluster import DBSCAN as SklearnDBSCAN  # type: ignore[assignment]
    from sklearn.ensemble import IsolationForest  # type: ignore[assignment]

    NUMPY_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    np = None
    pd = None
    PCA = None
    SklearnDBSCAN = None
    IsolationForest = None

try:  # pragma: no cover - optional dependency
    import cupy as cp  # type: ignore[assignment]
    from cuml.cluster import DBSCAN as CuMLDBSCAN  # type: ignore[assignment]

    GPU_CLUSTERING_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    cp = None
    CuMLDBSCAN = None

try:  # pragma: no cover - optional dependency
    from qdrant_client import QdrantClient  # type: ignore[assignment]
    from qdrant_client.http import models as qdrant_models  # type: ignore[assignment]
    from qdrant_client.models import PointIdsList  # type: ignore[assignment]
    try:
        from qdrant_client.models import OptimizersConfigDiff  # type: ignore[assignment]
    except ImportError:  # pragma: no cover - optional dependency
        OptimizersConfigDiff = None
except ImportError:  # pragma: no cover - optional dependency
    QdrantClient = None
    qdrant_models = None
    PointIdsList = None
    OptimizersConfigDiff = None

__all__ = [
    "NUMPY_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "GPU_CLUSTERING_AVAILABLE",
    "np",
    "pd",
    "PCA",
    "SklearnDBSCAN",
    "IsolationForest",
    "CuMLDBSCAN",
    "cp",
    "QdrantClient",
    "qdrant_models",
    "PointIdsList",
    "OptimizersConfigDiff",
]
