#!/usr/bin/env python
"""Extract per-residue ProtX embeddings and cache them to HDF5.

Run once per dataset split; all downstream experiments read from the cache.

Usage
-----
    python scripts/extract_embeddings.py \\
        --sequences data/raw/deeploc/train.fasta \\
        --model /path/to/protx_checkpoint \\
        --output data/embeddings/deeploc_train.h5 \\
        --batch-size 4 \\
        --device cuda

ProtX is distributed as a HuggingFace T5EncoderModel checkpoint.
If your course provides a different loading interface, adapt `load_model()`
and `embed_batch()` below — the rest of the script is format-agnostic.
"""
import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
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
    """Load ProtX tokenizer and encoder.  Model is frozen and set to eval."""
    from transformers import T5EncoderModel, T5Tokenizer

    log.info("Loading tokenizer from %s", model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)

    log.info("Loading model from %s", model_path)
    model = T5EncoderModel.from_pretrained(model_path)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


@torch.inference_mode()
def embed_batch(
    sequences: list[str],
    tokenizer,
    model,
    device: str,
) -> list[torch.Tensor]:
    """Return a list of [L, d] float32 CPU tensors, one per input sequence.

    ProtT5 / ProtX expect sequences as space-separated amino-acid characters.
    Special tokens (BOS/EOS) are stripped so that output length == input length.
    """
    spaced = [" ".join(list(s)) for s in sequences]
    enc = tokenizer(
        spaced,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    hidden = model(**enc).last_hidden_state   # [B, L_padded+special, d]

    embeddings: list[torch.Tensor] = []
    for i, seq in enumerate(sequences):
        L = len(seq)
        # ProtT5 prepends no BOS; it appends one EOS — take first L positions.
        emb = hidden[i, :L, :].cpu().float()
        embeddings.append(emb)
    return embeddings


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
                        help="Output HDF5 file.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Sequences per forward pass (reduce on OOM).")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(args.model, args.device)
    log.info("Device: %s | Hidden dim: %d", args.device,
             model.config.d_model)

    records = parse_fasta(args.sequences)
    log.info("Sequences to embed: %d", len(records))

    import numpy as np

    with EmbeddingStore(args.output, mode="w") as store:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            ids, seqs = zip(*batch)

            try:
                embeddings = embed_batch(list(seqs), tokenizer, model, args.device)
            except torch.cuda.OutOfMemoryError:
                log.warning("OOM on batch starting at %d — falling back to single", start)
                embeddings = []
                for seq in seqs:
                    embeddings.extend(
                        embed_batch([seq], tokenizer, model, args.device)
                    )

            for seq_id, emb in zip(ids, embeddings):
                store.write(seq_id, emb.numpy())

            done = min(start + args.batch_size, len(records))
            if done % 200 == 0 or done == len(records):
                log.info("  %d / %d sequences written", done, len(records))

    log.info("Done → %s", args.output)


if __name__ == "__main__":
    main()
