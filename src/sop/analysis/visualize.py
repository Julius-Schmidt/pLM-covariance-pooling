"""Target deliverables (project description §6.2):

* covariance matrix heatmap for a representative protein
* top-(i, j) co-activation analysis on a sample protein
* t-SNE / UMAP of pooled embeddings, coloured by class

These are visualisation helpers; experiment notebooks call them with already-
loaded poolers and embeddings.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..pooling.covariance import CovariancePooler


def covariance_heatmap(
    X: torch.Tensor,
    mask: torch.Tensor,
    pooler: CovariancePooler,
    *,
    title: str | None = None,
) -> plt.Figure:
    """Plot the dc × dc covariance embedding of one protein as a heatmap."""
    pooler.eval()
    with torch.no_grad():
        flat = pooler.pool(X.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
    dc = pooler.dc
    C = flat.reshape(dc, dc).cpu().numpy()

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(C, cmap="coolwarm", aspect="equal")
    ax.set_xlabel("R-projected feature j")
    ax.set_ylabel("L-projected feature i")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def top_coactivation_residues(
    X: torch.Tensor,
    mask: torch.Tensor,
    pooler: CovariancePooler,
    probe_weight_row: torch.Tensor,
    *,
    top_k: int = 5,
) -> list[tuple[int, int, list[tuple[int, float]]]]:
    """Identify residues driving the top-weighted (i, j) entries of the covariance.

    Args:
        X:                 [L, d] embeddings of a single protein.
        mask:              [L] bool, True for valid positions.
        pooler:            A trained CovariancePooler.
        probe_weight_row:  [dc²] absolute weight magnitudes from the probe head
                           (e.g. `model.probe.net[0].weight[class_idx, d:].abs()`
                           when interpreting the cov-portion of a hybrid embedding,
                           or `model.probe.net[0].weight[class_idx].abs()` for a
                           pure covariance probe).
        top_k:             Number of (i, j) entries and per-(i, j) residues to
                           return.

    Returns:
        For each of the top-K (i, j) entries, a tuple
        ``(i, j, [(residue_idx, contribution), ...])`` listing the residues with
        the largest ``(XL)_{k,i} · (XR)_{k,j}`` products.
    """
    pooler.eval()
    Xv = X[mask.bool()]
    with torch.no_grad():
        XL = pooler.proj_l(Xv)            # [L_valid, dc]
        XR = pooler.proj_r(Xv)            # [L_valid, dc]

    dc = pooler.dc
    weights = probe_weight_row.detach().cpu().reshape(dc, dc)
    # Top entries by absolute weight magnitude.
    flat_weights = weights.abs().flatten()
    top_idx = torch.topk(flat_weights, top_k).indices.tolist()

    results: list[tuple[int, int, list[tuple[int, float]]]] = []
    valid_indices = mask.nonzero(as_tuple=False).flatten().tolist()

    for flat_i in top_idx:
        i, j = divmod(flat_i, dc)
        contributions = (XL[:, i] * XR[:, j]).cpu()
        order = torch.topk(contributions.abs(), min(top_k, contributions.numel())).indices
        residue_hits = [(valid_indices[k.item()], float(contributions[k])) for k in order]
        results.append((i, j, residue_hits))
    return results


def tsne_or_umap(
    embeddings: np.ndarray,
    labels: np.ndarray | list,
    *,
    method: str = "tsne",
    title: str | None = None,
) -> plt.Figure:
    """Project ``embeddings`` to 2D and scatter, coloured by label.

    method = 'tsne' uses sklearn.manifold.TSNE; method = 'umap' requires the
    optional ``umap-learn`` package.
    """
    if method == "tsne":
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, init="pca", random_state=0).fit_transform(embeddings)
    elif method == "umap":
        import umap
        coords = umap.UMAP(n_components=2, random_state=0).fit_transform(embeddings)
    else:
        raise ValueError("method must be 'tsne' or 'umap'")

    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    for lbl in np.unique(labels):
        sel = labels == lbl
        ax.scatter(coords[sel, 0], coords[sel, 1], s=8, alpha=0.7, label=str(lbl))
    ax.legend(fontsize=7, markerscale=1.5, loc="best")
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
