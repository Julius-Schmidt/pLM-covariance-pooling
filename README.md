# Second-Order Pooling for Protein Language Models

**PP1 SoSe2026 · TU Munich · Chair of Bioinformatics I12**

Team: Joel Simon · Julius Schmidt · Lisa Börner · Andreas Weitz

---

## Research Question

Does compressing per-residue ProtX embeddings via **covariance pooling** improve downstream task performance compared to mean pooling, and at what bottleneck dimension does it become parameter-efficient?

## Method

Given per-residue embeddings **X** ∈ ℝ^{L×d} from frozen ProtX:

| Method | Formula | Output dim | Trainable params |
|---|---|---|---|
| Mean (baseline) | μ = (1/L) Xᵀ𝟏 | d | 0 |
| Covariance, supervised | C = (1/L)(XL)ᵀ(XR), L, R trained end-to-end | dc² | 2·d·dc |
| Covariance, unsupervised | C = (1/L)(XL)ᵀ(XR), L, R fitted by Frobenius reconstruction of XᵀX, then frozen | dc² | 2·d·dc (frozen) |
| Hybrid | [μ ; flat(C)] | d + dc² | 2·d·dc |

L, R ∈ ℝ^{d×dc} are two independent learnable projections (so C is asymmetric in general). The unsupervised regime trains them by minimising
‖XᵀX − L (XL)ᵀ(XR) Rᵀ‖²_F using the Frobenius equivalence ‖XᵀX‖_F = ‖XXᵀ‖_F so that no per-protein d×d matrix is materialised.

Every method feeds its pooled vector to the **same FNN probe head**, so comparisons are apples-to-apples.

## Project Structure

```
├── src/sop/
│   ├── pooling/
│   │   ├── base.py            # Pooler(nn.Module) interface
│   │   ├── mean.py            # MeanPooler
│   │   ├── covariance.py      # CovariancePooler — two learnable projections L, R
│   │   └── hybrid.py          # HybridPooler — [μ ; flat(C)] concat
│   ├── unsupervised/
│   │   └── frobenius_trainer.py   # autoencoder for ‖XᵀX − L C̃ Rᵀ‖²_F
│   ├── probes/
│   │   ├── fnn.py             # ProbeFNN (1 hidden layer, ReLU + dropout)
│   │   ├── model.py           # PoolingProbeModel = pooler + probe
│   │   ├── dataset.py         # ProteinEmbeddingDataset + collate_pad
│   │   ├── train_loop.py      # generic torch train loop (CE / MSE)
│   │   └── metrics.py         # accuracy, Spearman R
│   ├── data/store.py          # HDF5 embedding store
│   ├── utils/masking.py       # make_mask, apply_mask
│   └── analysis/              # aggregation + plots + cov visualisations
├── scripts/
│   ├── extract_embeddings.py     # ProtX → HDF5 (supports --layers for sweep)
│   ├── train_unsupervised_pool.py# fit + freeze the autoencoder
│   └── run_experiment.py         # train pooler + probe → JSON in results/runs/
├── configs/
│   ├── scl/{mean,cov_supervised,cov_unsupervised,hybrid}.yaml
│   └── meltome/{mean,cov_supervised,cov_unsupervised,hybrid}.yaml
├── tests/                     # pytest suite (masking invariance + correctness)
└── data/
    ├── raw/{deeploc,meltome}/  # FASTA + label CSVs (not tracked)
    └── embeddings/             # HDF5 caches (not tracked)
```

## Setup

```bash
conda env create -f environment.yml
conda activate sop
pip install -e .
```

> **GPU note:** edit [environment.yml](environment.yml) to swap `pytorch-cuda` for `cpuonly` if running without a GPU. Don't have both active at once.

ProtX inference is GPU-heavy; we run it on Google Colab and download the resulting HDF5 caches into `data/embeddings/`. All downstream training (probe head, autoencoder, analysis) runs locally on the cached tensors.

## Workflow

### 1 · Extract per-residue embeddings (once per split, on Colab)

```bash
python scripts/extract_embeddings.py \
    --sequences data/raw/deeploc/train.fasta \
    --model /path/to/protx_checkpoint \
    --output data/embeddings/deeploc_train.h5 \
    --batch-size 4 --device cuda
```

For the layer sweep, request multiple hidden states in one pass — each one becomes its own H5:

```bash
python scripts/extract_embeddings.py \
    --sequences data/raw/deeploc/train.fasta \
    --model /path/to/protx_checkpoint \
    --output data/embeddings/deeploc_train.h5 \
    --layers last 4 12 24 \
    --batch-size 4 --device cuda
```

### 2 · (Optional) Fit the unsupervised autoencoder once

```bash
python scripts/train_unsupervised_pool.py \
    --embeddings data/embeddings/deeploc_train.h5 \
    --d 1024 --dc 32 \
    --epochs 5 --batch-size 32 --lr 1e-3 \
    --output models/unsup_pooler_dc32.pt
```

The resulting checkpoint is reusable across tasks — point any `cov_unsupervised.yaml` config at it via `pretrained_path`.

### 3 · Run experiments

```bash
# Mean pooling baseline
python scripts/run_experiment.py --config configs/scl/mean.yaml

# Supervised covariance (L, R trained with the probe)
python scripts/run_experiment.py --config configs/scl/cov_supervised.yaml

# Frozen unsupervised covariance (loads pretrained_path)
python scripts/run_experiment.py --config configs/scl/cov_unsupervised.yaml

# Hybrid [μ ; flat(C)]
python scripts/run_experiment.py --config configs/scl/hybrid.yaml

# dc sweep
python scripts/run_experiment.py \
    --config configs/scl/cov_supervised.yaml \
    --dc 8 16 24 32 48
```

Results land under `results/runs/` as one JSON per (config, dc) with per-seed and aggregated metrics.

### 4 · Tests

```bash
pytest
```

Covers masking invariance (mean / covariance / hybrid), the Frobenius reconstruction identity, gradient flow through the supervised pooler, and the probe train-loop on both classification and regression.

## Label CSV format

Both `train_labels.csv` and `test_labels.csv` need a header row and two columns:

```
id,label
P12345,nucleus
Q67890,cytoplasm
```

For Meltome, `label` is a floating-point melting temperature (°C).

## Experiments

| Config | Task | Metric |
|---|---|---|
| `configs/scl/{mean,cov_supervised,cov_unsupervised,hybrid}.yaml` | Subcellular localisation (10-class) | Accuracy |
| `configs/meltome/{mean,cov_supervised,cov_unsupervised,hybrid}.yaml` | Thermostability (regression) | Spearman R |

Core grid: 4 pooling methods × 2 tasks × 3 seeds = 24 runs. dc sweep: dc ∈ {8, 16, 24, 32, 48} on both tasks. Layer sweep: re-run with embeddings from different ProtX layers.
