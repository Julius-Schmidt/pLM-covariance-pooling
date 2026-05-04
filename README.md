# Second-Order Pooling for Protein Language Models

**PP1 SoSe2026 · TU Munich · Chair of Bioinformatics I12**

Team: Joel Simon · Julius Schmidt · Lisa Börner · Andreas Weitz

---

## Research Question

Does compressing per-residue ProtX embeddings via **covariance pooling** improve downstream task performance compared to mean pooling, and at what bottleneck dimension does it become parameter-efficient?

## Method

Given per-residue embeddings **X** ∈ ℝ^{L×d} from frozen ProtX:

| Method | Formula | Output dim |
|---|---|---|
| Mean pooling (baseline) | μ = (1/L) Xᵀ1 | d |
| Covariance pooling | C = (1/L)(XU)ᵀ(XU), U from PCA | dc² |

U ∈ ℝ^{d×dc} contains the top-dc eigenvectors of the dataset-wide residue covariance **Σ = E[(x−μ)(x−μ)ᵀ]**, fitted once on training data and reused across tasks.

## Project Structure

```
├── src/sop/
│   ├── pooling/
│   │   ├── base.py          # Abstract Pooler interface
│   │   ├── mean.py          # MeanPooler
│   │   └── covariance.py    # CovariancePooler (PCA-based)
│   ├── data/
│   │   └── store.py         # HDF5 embedding store
│   └── utils/
│       └── masking.py       # make_mask, apply_mask
├── scripts/
│   ├── extract_embeddings.py   # ProtX → HDF5
│   └── run_experiment.py       # pool → probe → metrics
├── configs/                    # One YAML per (task, method) combination
├── tests/                      # Pytest suite (masking invariance + unit tests)
└── data/
    ├── raw/{deeploc,meltome}/  # Sequences + label CSVs (not tracked)
    └── embeddings/             # HDF5 caches (not tracked)
```

## Setup

```bash
conda env create -f environment.yml
conda activate sop
pip install -e .
```

> **GPU note:** edit `environment.yml` to match your CUDA version, or swap
> `pytorch-cuda` for `cpuonly` if running on CPU.

## Workflow

### Step 1 — Extract per-residue embeddings (once per split)

```bash
python scripts/extract_embeddings.py \
    --sequences data/raw/deeploc/train.fasta \
    --model /path/to/protx_checkpoint \
    --output data/embeddings/deeploc_train.h5 \
    --batch-size 4 \
    --device cuda
```

Repeat for each split (`train`, `test`) and each dataset (`deeploc`, `meltome`).

### Step 2 — Run experiment

```bash
# Mean pooling baseline
python scripts/run_experiment.py --config configs/deeploc_mean.yaml

# Covariance pooling (fits PCA projection on first run, caches to models/)
python scripts/run_experiment.py --config configs/deeploc_cov_dc32.yaml

# dc sweep (8, 16, 24, 32, 48)
python scripts/run_experiment.py \
    --config configs/deeploc_cov_dc32.yaml \
    --dc 8 16 24 32 48
```

Results are written to `results/` as JSON files.

### Step 3 — Run tests

```bash
pytest
```

The test suite verifies the critical **masking invariance**: adding zero-padded rows must never change the pooled result.

## Label CSV format

Both `train_labels.csv` and `test_labels.csv` must have a header row and two columns:

```
id,label
P12345,nucleus
Q67890,cytoplasm
```

For Meltome, `label` is a floating-point melting temperature (°C).

## Experiments

| Config | Task | Metric |
|---|---|---|
| `deeploc_{mean,cov_dcN}.yaml` | Subcellular localisation (10-class) | Accuracy |
| `meltome_{mean,cov_dcN}.yaml` | Thermostability (regression) | Spearman R |

Core results: 4 pooling methods × 2 tasks × 3 seeds = 24 runs (see project description).
dc sweep: dc ∈ {8, 16, 24, 32, 48}, 2 tasks × 3 seeds each.
