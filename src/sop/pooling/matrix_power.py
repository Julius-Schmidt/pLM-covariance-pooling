"""Matrix power normalisation of a covariance pool (iSQRT-COV).

Implements the differentiable matrix square root ``C -> C^{1/2}`` via the
coupled Newton-Schulz iteration of Li et al. (2018, CVPR). Applying it to the
covariance pool before flattening flattens the heavy-tailed eigenvalue
spectrum and maps the SPD matrix towards its Log-Euclidean tangent space, so a
linear probe sees a much better-conditioned input.

See ``docs/matrix_power_normalisation.md`` for the full derivation and
references.
"""
from __future__ import annotations

import torch


def isqrt_cov(C: torch.Tensor, n_iter: int = 5, eps: float = 1e-5) -> torch.Tensor:
    """Differentiable matrix square root via Newton-Schulz (Li et al. 2018).

    The power is taken in the **eigenbasis** of ``C`` (not entry-wise), so PSD
    structure is preserved and the map is a smooth bijection of the SPD manifold
    onto itself. No eigendecomposition is used — only ``2 * n_iter`` matmuls per
    matrix — so the backward pass is stable even with near-degenerate spectra.

    Args:
        C: ``[..., d, d]`` symmetric matrix (batched). For the asymmetric
           supervised pool the symmetric part is used (safety net below).
        n_iter: Number of Newton-Schulz steps (5 is the published default).
        eps: Numerical floor for the trace pre-normalisation.

    Returns:
        ``C^{1/2}`` of the same shape and dtype as ``C``.
    """
    # Newton-Schulz matmuls are numerically delicate; run in float32 even when
    # the caller is under fp16 autocast, then cast back.
    orig_dtype = C.dtype
    C = C.float()

    # Symmetrise. For a tied/PCA pool C is already symmetric PSD and this is a
    # no-op; for the asymmetric supervised pool it projects onto the symmetric
    # part so the iteration is well-defined (docs §3.2).
    C = 0.5 * (C + C.transpose(-2, -1))

    d = C.shape[-1]
    I = torch.eye(d, device=C.device, dtype=C.dtype).expand_as(C)

    # 1. Trace-normalise so ||A|| < 1 (Newton-Schulz convergence condition).
    trace = C.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    trace = trace.clamp_min(eps)

    # Ridge-regularise: lift the spectrum off zero so rank-deficient covariances
    # (dc > #residues, or collinear features) stay positive definite. This keeps
    # Newton-Schulz convergent and separates the clustered eigenvalues that
    # otherwise make the eigh backward (1/(λi-λj)) blow up to NaN.
    A = C / trace + eps * I

    # 2. Coupled Newton-Schulz: Y_k -> A^{1/2}, Z_k -> A^{-1/2}.
    Y, Z = A, I.clone()
    for _ in range(n_iter):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    # 3. Restore the trace scale: C^{1/2} = sqrt(tr C) * A^{1/2}.
    C_sqrt = Y * trace.sqrt()

    if torch.isfinite(C_sqrt).all():
        return C_sqrt.to(orig_dtype)

    # Robust fallback (docs §3.1, "Option A"). Newton-Schulz only converges for
    # PSD inputs; the asymmetric supervised pool can symmetrise to an indefinite
    # matrix where the iteration diverges. There we take the exact eigenbasis
    # square root with eigenvalues clamped to >= 0 (PSD projection). The ridge
    # keeps the spectrum non-degenerate; cuSOLVER still occasionally fails to
    # converge on the GPU, so retry on CPU (LAPACK syevd is more tolerant) and
    # scrub any residual non-finite entries so a single bad batch cannot abort
    # the whole run.
    A_ridge = C + eps * trace * I
    try:
        evals, evecs = torch.linalg.eigh(A_ridge)
    except torch._C._LinAlgError:
        evals, evecs = torch.linalg.eigh(A_ridge.cpu())
        evals, evecs = evals.to(C.device), evecs.to(C.device)
    roots = evals.clamp_min(eps).sqrt()                      # [..., d]
    C_sqrt = (evecs * roots.unsqueeze(-2)) @ evecs.transpose(-2, -1)
    C_sqrt = torch.nan_to_num(C_sqrt)
    return C_sqrt.to(orig_dtype)
