#!/usr/bin/env bash
# Full end-to-end grid runner for g4dn.xlarge:
#   1. Move embeddings to local NVMe (g4dn instance store) for ~5× read speedup
#   2. Fit PCA + unsupervised poolers for dc ∈ {8, 16, 24, 32, 48}
#   3. Run main grid (10 configs at dc=32)
#   4. Run dc sweep for the 4 covariance methods × {8, 16, 24, 48} × 2 tasks
#
# Assumes: venv already exists at .venv, data already in data/embeddings + data/raw.
# Run from the repo root inside tmux:
#     tmux new -s sop
#     bash scripts/run_full_grid_aws.sh 2>&1 | tee run_full.log

set -euo pipefail

VENV_DIR=".venv"
DCS=(8 16 24 32 48)
NVME_MOUNT="/mnt/nvme"

log() { printf "\n\033[1;34m[%(%H:%M:%S)T]\033[0m %s\n" -1 "$*"; }

# --- 0. Activate venv -------------------------------------------------------
log "Activating venv"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python --version

# --- 1. NVMe setup ----------------------------------------------------------
# g4dn.xlarge has a 125 GB instance store NVMe (/dev/nvme1n1 typically).
# Format + mount it once. Survives reboot only if you remount manually —
# instance-store data is LOST on stop/terminate.
log "NVMe setup"
if [[ ! -d "$NVME_MOUNT" || -z "$(ls -A "$NVME_MOUNT" 2>/dev/null)" ]]; then
    NVME_DEV=$(lsblk -dn -o NAME,SIZE,MOUNTPOINT | awk '$3=="" && $1 ~ /^nvme[1-9]/ {print $1; exit}')
    if [[ -z "$NVME_DEV" ]]; then
        echo "No unmounted NVMe device found. lsblk output:" >&2
        lsblk
        exit 1
    fi
    log "Formatting /dev/$NVME_DEV as ext4"
    sudo mkfs.ext4 -F "/dev/$NVME_DEV"
    sudo mkdir -p "$NVME_MOUNT"
    sudo mount "/dev/$NVME_DEV" "$NVME_MOUNT"
    sudo chown "$USER:$USER" "$NVME_MOUNT"
fi
df -h "$NVME_MOUNT"

# --- 2. Copy embeddings to NVMe and symlink ---------------------------------
log "Copy embeddings to NVMe (one-time, ~5–10 min for ~72 GB)"
mkdir -p "$NVME_MOUNT/embeddings"
for h5 in data/embeddings/*.h5; do
    base=$(basename "$h5")
    target="$NVME_MOUNT/embeddings/$base"
    if [[ ! -f "$target" ]]; then
        log "  copying $base"
        cp "$h5" "$target"
    else
        log "  $base already on NVMe — skipping"
    fi
done

# Replace data/embeddings/*.h5 with symlinks pointing at NVMe so the configs
# keep working unchanged.
log "Symlink data/embeddings/*.h5 → NVMe"
for h5 in data/embeddings/*.h5; do
    base=$(basename "$h5")
    if [[ ! -L "$h5" ]]; then
        rm -f "$h5"
        ln -s "$NVME_MOUNT/embeddings/$base" "$h5"
    fi
done
ls -lh data/embeddings/

# --- 3. Fit poolers for every dc value -------------------------------------
log "Fit PCA + unsupervised poolers for dc ∈ {${DCS[*]}}"
mkdir -p models
for DC in "${DCS[@]}"; do
    pca_out="models/pca_pooler_dc${DC}.pt"
    unsup_out="models/unsup_pooler_dc${DC}.pt"

    if [[ ! -f "$pca_out" ]]; then
        log "  fit PCA dc=$DC"
        python scripts/fit_pca_pool.py \
            --embeddings data/embeddings/deeploc_train.h5 \
            --d 1024 --dc "$DC" \
            --output "$pca_out"
    fi
    if [[ ! -f "$unsup_out" ]]; then
        log "  fit unsupervised dc=$DC"
        python scripts/train_unsupervised_pool.py \
            --embeddings data/embeddings/deeploc_train.h5 \
            --d 1024 --dc "$DC" \
            --epochs 5 --batch-size 32 --lr 1e-3 \
            --output "$unsup_out"
    fi
done
ls -lh models/

# --- 4. Main grid (10 configs at dc=32) ------------------------------------
main_configs=(
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

log "Main grid: ${#main_configs[@]} configs × 3 seeds @ dc=32"
for cfg in "${main_configs[@]}"; do
    log "→ MAIN  $cfg"
    python scripts/run_experiment_aws.py --config "$cfg"
done

# --- 5. dc sweep (cov methods only, mean has no dc) ------------------------
sweep_configs=(
    configs/scl/cov_supervised.yaml
    configs/scl/cov_unsupervised.yaml
    configs/scl/cov_pca.yaml
    configs/scl/hybrid.yaml
    configs/meltome/cov_supervised.yaml
    configs/meltome/cov_unsupervised.yaml
    configs/meltome/cov_pca.yaml
    configs/meltome/hybrid.yaml
)

# Sweep ALL dc values — the dc=32 run produced in step 4 will be overwritten
# with a fresh run, which is fine (same code, same data, deterministic seeds).
log "dc sweep: ${#sweep_configs[@]} configs × 5 dc values = $((${#sweep_configs[@]} * 5)) runs"
for cfg in "${sweep_configs[@]}"; do
    log "→ SWEEP $cfg"
    python scripts/run_experiment_aws.py --config "$cfg" --dc "${DCS[@]}"
done

log "ALL DONE. Results in results/runs/."
log "Pull them down with:"
echo "  scp -r ubuntu@<instance>:$(pwd)/results ./"
