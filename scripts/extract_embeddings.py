#!/usr/bin/env python
"""Extract per-residue ProtX embeddings and cache them to HDF5.

Run once per dataset split. All downstream experiments read from the cache.

Usage
-----
    # Last layer only (default — what every experiment uses by default)
    python scripts/extract_embeddings.py \\
        --sequences data/raw/deeploc/train.fasta \\
        --model /path/to/protx_checkpoint \\
        --output data/embeddings/deeploc_train.h5 \\
        --batch-size 4 --device cuda

    # Layer sweep — extract multiple hidden states in one pass.
    # Outputs one HDF5 per layer: deeploc_train_layer{N}.h5.
    python scripts/extract_embeddings.py \\
        --sequences data/raw/deeploc/train.fasta \\
        --model /path/to/protx_checkpoint \\
        --output data/embeddings/deeploc_train.h5 \\
        --layers last 4 12 24 \\
        --batch-size 4 --device cuda

ProtX is distributed as a HuggingFace T5EncoderModel checkpoint. If your
course provides a different loading interface, adapt ``load_model()`` and
``embed_batch()`` below — the rest of the script is format-agnostic.
"""
import argparse
import logging
from pathlib import Path

import torch

from sop.data.store import EmbeddingStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Return [(seq_id, sequence), ...]. ID is truncated at first whitespace."""
    records: list[tuple[str, str]] = []
    seq_id: str | None = None
    parts: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    records.append((seq_id, "".join(parts)))
                seq_id = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
    if seq_id is not None:
        records.append((seq_id, "".join(parts)))
    return records


# ---------------------------------------------------------------------------
# Model loading & inference (ProtX / HuggingFace T5EncoderModel)
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: str):
    """Load ProtX tokenizer and encoder. Model is frozen and set to eval."""
    from transformers import T5EncoderModel, T5Tokenizer

    log.info("Loading tokenizer from %s", model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)

    log.info("Loading model from %s", model_path)
    model = T5EncoderModel.from_pretrained(model_path)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


def resolve_layers(layer_args: list[str], n_layers: int) -> list[int]:
    """Resolve user-friendly layer specifiers to absolute hidden-state indices.

    HuggingFace ``hidden_states`` is a tuple of length ``n_layers + 1``:
    index 0 is the embedding output, index ``n_layers`` is the final layer.

    Accepts:
      * 'last'              → final layer (n_layers)
      * 'middle'            → n_layers // 2
      * 'embed'             → 0
      * any integer string  → taken verbatim, with negatives counting from the end
    """
    out: list[int] = []
    for spec in layer_args:
        if spec == "last":
            out.append(n_layers)
        elif spec == "middle":
            out.append(n_layers // 2)
        elif spec == "embed":
            out.append(0)
        else:
            i = int(spec)
            if i < 0:
                i = n_layers + 1 + i
            out.append(i)
    return out


@torch.inference_mode()
def embed_batch(
    sequences: list[str],
    tokenizer,
    model,
    device: str,
    layer_indices: list[int] | None,
) -> dict[int | None, list[torch.Tensor]]:
    """Forward a batch through ProtX and return per-layer per-protein tensors.

    Returns a dict mapping layer index → list of [L, d] CPU float32 tensors,
    one per input sequence. If ``layer_indices`` is None, the dict has a
    single key ``None`` with the last-hidden-state tensors.
    """
    spaced = [" ".join(list(s)) for s in sequences]
    enc = tokenizer(
        spaced,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    if layer_indices is None:
        out = model(**enc)
        hidden_per_layer = {None: out.last_hidden_state}
    else:
        out = model(**enc, output_hidden_states=True)
        hidden_per_layer = {idx: out.hidden_states[idx] for idx in layer_indices}

    result: dict[int | None, list[torch.Tensor]] = {k: [] for k in hidden_per_layer}
    for layer_key, hidden in hidden_per_layer.items():
        for i, seq in enumerate(sequences):
            L = len(seq)
            # ProtT5/ProtX prepends no BOS; appends one EOS — take first L positions.
            emb = hidden[i, :L, :].cpu().float()
            result[layer_key].append(emb)
    return result


def output_path_for_layer(base: Path, layer_idx: int | None) -> Path:
    """Append `_layer{N}` to the base output stem when extracting multiple layers."""
    if layer_idx is None:
        return base
    return base.with_name(f"{base.stem}_layer{layer_idx}{base.suffix}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True,
                        help="Input FASTA file.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path or HuggingFace ID of the ProtX checkpoint.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output HDF5 file (becomes a stem if --layers used).")
    parser.add_argument("--layers", nargs="*", default=None,
                        help="Layer specifiers (e.g. 'last 12 24 embed'). "
                             "If omitted, only the last hidden state is saved.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Sequences per forward pass (reduce on OOM).")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(args.model, args.device)
    n_layers = model.config.num_layers
    log.info("Device: %s | Hidden dim: %d | Layers: %d",
             args.device, model.config.d_model, n_layers)

    layer_indices = (
        resolve_layers(args.layers, n_layers) if args.layers else None
    )
    if layer_indices is not None:
        log.info("Extracting layers: %s", layer_indices)

    records = parse_fasta(args.sequences)
    log.info("Sequences to embed: %d", len(records))

    # One EmbeddingStore per layer (or one for the default last-hidden-state run).
    keys = [None] if layer_indices is None else layer_indices
    stores = {
        k: EmbeddingStore(output_path_for_layer(args.output, k), mode="w").__enter__()
        for k in keys
    }
    try:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            ids, seqs = zip(*batch)

            try:
                per_layer = embed_batch(
                    list(seqs), tokenizer, model, args.device, layer_indices,
                )
            except torch.cuda.OutOfMemoryError:
                log.warning("OOM on batch starting at %d — falling back to single", start)
                per_layer = {k: [] for k in keys}
                for seq in seqs:
                    one = embed_batch(
                        [seq], tokenizer, model, args.device, layer_indices,
                    )
                    for k, embs in one.items():
                        per_layer[k].extend(embs)

            for k, embs in per_layer.items():
                for seq_id, emb in zip(ids, embs):
                    stores[k].write(seq_id, emb.numpy())

            done = min(start + args.batch_size, len(records))
            if done % 200 == 0 or done == len(records):
                log.info("  %d / %d sequences written", done, len(records))
    finally:
        for s in stores.values():
            s.__exit__(None, None, None)

    for k in keys:
        log.info("Done → %s", output_path_for_layer(args.output, k))


if __name__ == "__main__":
    main()
