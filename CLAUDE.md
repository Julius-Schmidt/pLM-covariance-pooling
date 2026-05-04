# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PP1 SoSe2026 (TU Munich, Chair of Bioinformatics I12). Tests whether **second-order (covariance) pooling** of frozen ProtX per-residue embeddings beats mean pooling on per-protein downstream tasks (subcellular localisation classification + Meltome thermostability regression). See [README.md](README.md) and [project-2-explanation.md](project-2-explanation.md) for the full motivation.

## Setup

```bash
conda env create -f environment.yml
conda activate sop
pip install -e .
```

`environment.yml` defaults to `cpuonly`; swap to `pytorch-cuda=12.1` for GPU. Embedding extraction is GPU-heavy and is run on Colab — local boxes only run probe training, autoencoder/PCA fitting, and analysis on the cached HDF5 tensors.

## Common commands

```bash
pytest                                                              # full test suite
pytest tests/test_covariance.py                                     # single file
pytest tests/test_covariance_pca.py::test_output_is_symmetric       # single test
pytest -k masking                                                   # tests matching keyword

python scripts/run_experiment.py --config configs/scl/mean.yaml     # one method, one task
python scripts/run_experiment.py --config configs/scl/cov_supervised.yaml --dc 8 16 24 32 48   # dc sweep
```

## Architecture

The pipeline is **strictly staged**: extract once → fit pooler (optional) → train probe. Every stage reads cached tensors from disk; nothing re-runs the frozen ProtX backbone.

### Five pooling methods, one interface

All poolers subclass [Pooler](src/sop/pooling/base.py) (`pool(X, mask) -> [B, embedding_dim]`) and feed into the **same** [ProbeFNN](src/sop/probes/fnn.py) head via [PoolingProbeModel](src/sop/probes/model.py). This is what makes the head-to-head comparison fair.

| Method | Module | Trainable params | Output dim | How L, R fit |
|---|---|---|---|---|
| `mean` | [mean.py](src/sop/pooling/mean.py) | 0 | d | — |
| `cov_supervised` | [covariance.py](src/sop/pooling/covariance.py) | 2·d·dc | dc² | end-to-end with probe |
| `cov_unsupervised` | [covariance.py](src/sop/pooling/covariance.py) (frozen) | 0 (frozen) | dc² | Frobenius autoencoder, SGD ([frobenius_trainer.py](src/sop/unsupervised/frobenius_trainer.py)) |
| `cov_pca` | [covariance_pca.py](src/sop/pooling/covariance_pca.py) | 0 (frozen) | dc² | top-dc eigenvectors of dataset Σ, closed-form |
| `hybrid` | [hybrid.py](src/sop/pooling/hybrid.py) | 2·d·dc | d + dc² | wraps a `CovariancePooler` (sup or unsup) |

`CovariancePooler` uses two **independent** projections L, R (asymmetric C). `CovariancePCAPooler` is the **symmetric tied-weights** special case (L = R = U from PCA → C is symmetric/PSD). PCA is the algorithmic baseline answering: does the autoencoder's extra freedom buy anything over the closed-form solution? See [test_covariance_pca.py:test_output_is_symmetric](tests/test_covariance_pca.py).

### Data flow

1. **Extract** ([extract_embeddings.py](scripts/extract_embeddings.py), Colab): FASTA → ProtX (HuggingFace `T5EncoderModel`) → HDF5 via [EmbeddingStore](src/sop/data/store.py). Per-protein groups `/{seq_id}/embeddings` plus `length` attr; first L positions are taken (ProtT5/ProtX appends one EOS, no BOS). `--layers last 4 12 24` writes one HDF5 per layer (`*_layer{N}.h5`) for the layer sweep.
2. **Fit frozen pooler (optional)**: [train_unsupervised_pool.py](scripts/train_unsupervised_pool.py) (SGD autoencoder) or [fit_pca_pool.py](scripts/fit_pca_pool.py) (closed-form, two streaming passes). Both save a `.pt` checkpoint that downstream configs reference via `pretrained_path`.
3. **Run experiment** ([run_experiment.py](scripts/run_experiment.py)): YAML → builds pooler + `ProbeFNN` → [train_probe](src/sop/probes/train_loop.py) over 3 seeds → JSON in `results/runs/`. Only `requires_grad=True` parameters are optimised, so frozen pooler weights stay frozen automatically.

### Critical invariants

- **Masking**: every pooler must zero padded rows *before* the bilinear/mean sum and divide by the **valid** length, not the padded length. The PCA pooler additionally centres *before* re-zeroing padding — order matters because centring turns originally-zero pad rows into `-μ`. Padding invariance is unit-tested per pooler ([test_masking.py](tests/test_masking.py), `test_padding_does_not_change_result` in each pooler test).
- **Frobenius reconstruction trick**: never materialise the d×d matrix XᵀX (d≈1024 → 1M entries). [frobenius_recon_loss](src/sop/unsupervised/frobenius_trainer.py) expands ‖XᵀX − L C̃ Rᵀ‖²_F into terms over the L×L Gram matrix and dc×dc matrices only. Loss is normalised by L² so long proteins don't dominate.
- **HDF5 store**: `EmbeddingStore` is opened lazily on first `__getitem__` and held open for the dataset's lifetime — safe for default single-worker DataLoader; set `persistent_workers=False` if you raise `num_workers`.
- **Label CSV format**: header `id,label`. Classification labels are strings (mapped to indices in `collate_pad`); regression labels are floats.

### Frozen-pooler workflow

`cov_unsupervised`, `cov_pca`, and (optionally) `hybrid` all load checkpoints via `from_pretrained`. `run_experiment.py` rewrites `pretrained_path` automatically when sweeping `--dc N1 N2 ...`: `models/foo.pt` → `models/foo_dc{N}.pt`. Fit one checkpoint per dc value before sweeping.

## Conventions

- Source layout uses `src/` (PEP 420 / setuptools `packages.find`). Import as `from sop.pooling.covariance import CovariancePooler`.
- Configs live under `configs/{scl,meltome}/{mean,cov_supervised,cov_unsupervised,cov_pca,hybrid}.yaml` — one file per (task, method).
- `data/embeddings/*.h5`, `models/*.pt`, `results/*.json` are gitignored.
