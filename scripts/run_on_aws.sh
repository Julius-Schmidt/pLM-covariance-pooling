#!/usr/bin/env bash
# End-to-end runner for an AWS g4dn.xlarge (NVIDIA T4) box.
#
# Assumes:
#   * Ubuntu 22.04 + "Deep Learning AMI GPU PyTorch" (nvidia driver already present)
#   * This repo already cloned and CWD is the repo root
#
# Usage:
#   ssh ubuntu@<instance>
#   git clone https://github.com/<user>/pLM-covariance-pooling.git
#   cd pLM-covariance-pooling
#   bash scripts/run_on_aws.sh
#
# When done, scp results back:
#   scp -r ubuntu@<instance>:~/pLM-covariance-pooling/results ./

set -euo pipefail

DRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1LZvQNYDQQzO5noWwtLS2sQsjLmgQXtYA"
VENV_DIR=".venv"
DC=32   # bottleneck dim for the pretrained PCA / unsupervised poolers

log() { printf "\n\033[1;34m[%(%H:%M:%S)T]\033[0m %s\n" -1 "$*"; }

# --- 0. Sanity check GPU ----------------------------------------------------
log "GPU check"
if ! command -v nvidia-smi >/dev/null; then
    echo "nvidia-smi not found. Use a Deep Learning AMI or install NVIDIA drivers." >&2
    exit 1
fi
nvidia-smi | head -n 20

# --- 1. Python env ----------------------------------------------------------
log "Python venv"
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel

log "PyTorch (CUDA 12.1 wheels — works with T4)"
pip install --index-url https://download.pytorch.org/whl/cu121 torch

log "Project + deps"
pip install -e .
pip install gdown

python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda available: {torch.cuda.is_available()}  device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

# --- 2. Pull data from Google Drive -----------------------------------------
log "Download data from Google Drive"
mkdir -p data
if [[ ! -d data/embeddings || -z "$(ls -A data/embeddings 2>/dev/null)" ]]; then
    rm -rf _drive_tmp
    gdown --folder "$DRIVE_FOLDER_URL" -O _drive_tmp
    # gdown places the Drive folder's contents under _drive_tmp/<folder-name>/
    DRIVE_ROOT="$(find _drive_tmp -maxdepth 2 -type d -name embeddings -printf '%h\n' | head -n1)"
    if [[ -z "$DRIVE_ROOT" ]]; then
        echo "Could not find an 'embeddings' subfolder in the downloaded Drive folder." >&2
        find _drive_tmp -maxdepth 3 -type d
        exit 1
    fi
    log "Drive root resolved to $DRIVE_ROOT"
    mkdir -p data/embeddings data/raw
    cp -r "$DRIVE_ROOT/embeddings/." data/embeddings/
    cp -r "$DRIVE_ROOT/raw/." data/raw/
    rm -rf _drive_tmp
else
    log "data/embeddings already populated — skipping download"
fi

log "Verify expected files"
required=(
    data/embeddings/deeploc_train.h5
    data/embeddings/deeploc_test.h5
    data/embeddings/meltome_train.h5
    data/embeddings/meltome_test.h5
    data/raw/deeploc/train_labels.csv
    data/raw/deeploc/test_labels.csv
    data/raw/meltome/train_labels.csv
    data/raw/meltome/test_labels.csv
)
missing=0
for f in "${required[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "  MISSING: $f"
        missing=1
    fi
done
if [[ $missing -ne 0 ]]; then
    echo
    echo "Some expected files are missing. Got these under data/:"
    find data -maxdepth 3 -type f | sort
    echo
    echo "Rename/move the Drive files to match the paths above, then re-run." >&2
    exit 1
fi

# --- 3. Fit frozen poolers (needed by cov_pca / cov_unsupervised) ----------
log "Fit PCA pooler (dc=$DC) — closed form, fast"
mkdir -p models
if [[ ! -f "models/pca_pooler_dc${DC}.pt" ]]; then
    python scripts/fit_pca_pool.py \
        --embeddings data/embeddings/deeploc_train.h5 \
        --d 1024 --dc "$DC" \
        --output "models/pca_pooler_dc${DC}.pt"
fi

log "Fit unsupervised (autoencoder) pooler (dc=$DC)"
if [[ ! -f "models/unsup_pooler_dc${DC}.pt" ]]; then
    python scripts/train_unsupervised_pool.py \
        --embeddings data/embeddings/deeploc_train.h5 \
        --d 1024 --dc "$DC" \
        --epochs 5 --batch-size 32 --lr 1e-3 \
        --output "models/unsup_pooler_dc${DC}.pt"
fi

# --- 4. Run the experiment grid --------------------------------------------
configs=(
    configs/scl/mean.yaml
    configs/scl/cov_supervised.yaml
    configs/scl/cov_unsupervised.yaml
    configs/scl/cov_pca.yaml
    configs/scl/hybrid.yaml
    configs/meltome/mean.yaml
    configs/meltome/cov_supervised.yaml
    configs/meltome/cov_unsupervised.yaml
    configs/meltome/cov_pca.yaml
    configs/meltome/hybrid.yaml
)

log "Running ${#configs[@]} configs × 3 seeds each"
for cfg in "${configs[@]}"; do
    log "→ $cfg"
    python scripts/run_experiment_aws.py --config "$cfg"
done

# --- 5. (Optional) dc sweep ------------------------------------------------
# Uncomment to also run the dc sweep on the supervised covariance variant.
#
# log "dc sweep — supervised covariance"
# for cfg in configs/scl/cov_supervised.yaml configs/meltome/cov_supervised.yaml; do
#     python scripts/run_experiment_aws.py --config "$cfg" --dc 8 16 24 32 48
# done

log "DONE. Results in results/runs/. Pull them down with:"
echo "  scp -r ubuntu@<instance>:$(pwd)/results ./"
