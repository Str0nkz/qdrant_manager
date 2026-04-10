# Qdrant Cleaner

A lean, open-source Qdrant collection cleaner focused on inspection, deduplication, vector repair, sparse-content detection, and targeted cleanup tasks.

## Installation

```bash
pip install -r requirements.txt
```

Or install directly from a repository:

```bash
pip install git+https://github.com/Str0nkz/qdrant_manager.git
```

## Package Surface

The public package surface is intentionally small:

- `qdrant_manager.cleaner` — synchronous collection cleaner built on `qdrant-client`
- `qdrant_manager.cleaner_cli` — interactive cleaner-only CLI
- `qdrant_manager.config` — OSS-safe config for the Qdrant endpoint and canonical text fields

## Configuration

The cleaner reads `QDRANT_URL` from the environment and defaults to `http://localhost:6333`.

```bash
export QDRANT_URL=http://localhost:6333
python -m qdrant_manager
```

## Launching the Cleaner CLI

```bash
python -m qdrant_manager
```

On startup you will be prompted for a Qdrant URL (pre-filled from `QDRANT_URL` or `http://localhost:6333`). The CLI checks connectivity immediately and shows a green **Connected** or red **Unreachable** status. If the instance is not reachable you can enter a different URL or continue anyway — the main menu stays up and reports errors per-action rather than crashing.

## Programmatic Usage

```python
from qdrant_manager import QdrantCleaner

cleaner = QdrantCleaner(host="localhost", port=6333)
cleaner.set_collection("my_collection")

summary = cleaner.analyze_collection(progress_callback=print)
result = cleaner.one_stop_clean(dry_run=True, progress_callback=print)
```

## CLI Features

The interactive CLI provides the following capabilities:

1. **List Collections** — Browse available collections with point counts
2. **Analyze Collection** — Health check with duplicate detection and vector metrics
3. **Clean Collection** — One-stop cleanup with duplicate removal
4. **Repair Vectors** — Normalize and fix invalid vectors (NaN, Inf, zero-norm, unnormalized)
5. **Semantic Deduplication** — DBSCAN-based semantic duplicate detection with GPU support
6. **Scan Sparse Content** — Identify low-quality or thin content using payload and vector signals
7. **Outlier Detection** — Find anomalous vectors with IsolationForest, DBSCAN, or statistical methods
8. **Remove Points by Payload Filter** — Delete points matching arbitrary `field=value` payload conditions

## GPU-Accelerated Clustering

Semantic deduplication and DBSCAN-based outlier detection can use GPU acceleration when RAPIDS cuML is installed and compatible with your CUDA runtime. The cleaner accepts a clustering backend of `auto`, `gpu`, or `cpu`:

- `auto` — prefers GPU DBSCAN and falls back to CPU automatically
- `gpu` — attempts the GPU path first, then falls back to CPU with a warning if the CUDA stack is not usable
- `cpu` — forces the scikit-learn implementation

GPU acceleration applies to DBSCAN-based workflows only. Pairwise deduplication, IsolationForest, and statistical outlier detection remain CPU-based.

### Installation for GPU Support

For CUDA 12 environments (e.g., RTX 4090/5090):

```bash
pip install cupy-cuda12x cuml-cu12
```

Adjust the package names to match your CUDA version.

## Scanning for Sparse / Low-Quality Content

Over time collections accumulate semantically thin content: bibliography sections, boilerplate headers, figure captions, and random short chunks. The **Scan Sparse Content** feature identifies these using two complementary signals combined into a single score.

### How Scoring Works

The scorer is schema-agnostic — it works with any Qdrant payload without requiring specific field names.

| Signal | How it is derived | Weight | Direction |
|---|---|---|---|
| Field density | Proportion of non-empty fields in the payload | 2.0 | inverted |
| Text length | Length of the primary text field, normalised to 2,000 chars | 1.5 | inverted |
| Quality score | First payload field whose name contains `score`, `quality`, `confidence`, `relevance`, or `rank` and holds a `[0, 1]` float | 2.0 | inverted |
| Nesting | Whether any payload field contains a nested dict (richer structure = denser) | 0.5 | inverted |

*Inverted* means a higher raw value produces a lower sparsity score (i.e. it is a good signal). Signals absent from a point are skipped and the remaining weights are re-normalised automatically, so the scorer degrades gracefully on minimal schemas.

> **Note:** the metadata heuristics reflect general assumptions about payload richness and may not suit every schema. The vector anomaly signal (IsolationForest) works for any collection regardless of payload shape. If the metadata score is not meaningful for your data, set `vector_weight=1.0` to rely on vector anomaly only, or fork `_score_payload_sparsity` and adjust the signals and weights to match your model.

The metadata composite is combined with a vector-space anomaly score from IsolationForest:

```
final_score = vector_weight x vec_anomaly + (1 - vector_weight) x metadata_score
```

`vec_anomaly` is derived from `IsolationForest.decision_function()` (continuous, not binary), flipped and min-max normalised to `[0, 1]`. A score of **1.0 = most sparse**, **0.0 = most dense**.

Points with no vector fall back to metadata score only; points with no metadata signal fall back to vector score only.

### Recommended Workflow for Periodic Maintenance

**Step 1 — report pass (safe, read-only)**
```
Option 6 -> Bottom percentile -> 0.10 -> vector weight 0.55 -> Report only
```
Browse the candidate table. Rows scoring >= 0.70 are strong candidates; >= 0.50 are worth reviewing. If both the `Vec` and `Meta` columns are high the point is almost certainly noise.

**Step 2 — tag pass (non-destructive)**
```
Option 6 -> Score threshold -> 0.65 -> Tag -> Dry run: No
```
This writes `_sparsity_score` to each candidate's Qdrant payload. You can then browse or filter directly in Qdrant's dashboard or API:
```json
{ "filter": { "must": [{ "key": "_sparsity_score", "range": { "gte": 0.65 } }] } }
```

**Step 3 — delete pass (destructive, confirm carefully)**
```
Option 6 -> Score threshold -> 0.70 -> Delete -> Dry run: No -> Confirm
```
After deletion, run **Semantic Deduplication** (option 5) — removing sparse chunks can expose near-duplicate pairs that were previously spread across good and bad content.

### Tuning Tips

- **Small chunks (<= 256 tokens)** produce more sparse vectors than large ones. Expect 10-20% of a heavily chunked PDF collection to score above 0.60 on a first scan.
- **Boilerplate content** (bibliography sections, figure captions, repeated headers) tends to be short, has low field density, and sits geometrically isolated in vector space — both metadata and vector signals will agree, so it reliably surfaces in the top percentile.
- **Legitimate niche content** (unusual topic, geometrically isolated in vector space) will score high on the vector anomaly signal but low on metadata sparsity. The combined score keeps it safe from deletion, which is the main reason to use both signals together.
- **Default vector weight of 0.55** gives a slight edge to the geometric signal. Lower it toward 0.30-0.40 if your payload metadata is sparse or inconsistent; raise it toward 0.70 if payloads are rich and reliable.
- **Quality score fields** — if your points carry a `quality_score`, `confidence`, or similarly named `[0, 1]` field the scorer will automatically detect and weight it at 2.0, making the metadata signal much stronger.
- The IsolationForest fit holds all vectors in RAM simultaneously. At 1536 dimensions expect roughly 6 MB per 1,000 points. On a 64 GB machine this is not a constraint for any realistic collection size.

### Programmatic Usage

```python
from qdrant_manager import QdrantCleaner

cleaner = QdrantCleaner(host="localhost", port=6333)
cleaner.set_collection("my_collection")

result = cleaner.scan_sparse(
    percentile=0.10,       # or: threshold=0.65 / top_n=500
    vector_weight=0.55,
    action="tag",          # "report" | "tag" | "delete"
    dry_run=False,
    progress_callback=print,
)

print(f"Candidates : {result['candidates']}")
print(f"Tagged     : {result['tagged']}")

# Top candidates with full breakdown
for c in result["top_candidates"][:10]:
    print(c["final_score"], c["title"], c["breakdown"])
```

## Optional Dependencies

Vector operations and PCA-based tooling require NumPy and scikit-learn. These are imported lazily via `dependencies.py`; when they are missing the package continues to work but vector-heavy workflows will not be available.
