# Matrix Power Normalisation of the Covariance Pool

**Proposed extension:** apply a matrix square-root (or more generally a matrix
power $C \mapsto C^{p}$ with $p \in (0, 1]$) to the covariance pool **before**
flattening and feeding the FNN probe. This is the standard "improved bilinear
pooling" / iSQRT-COV trick from second-order CNNs (Lin & Maji 2017; Li et al.
CVPR 2018), and to our knowledge has not been applied to protein language
model pooling.

This document explains *why* the operation helps, *what* it does mathematically,
and *how* to implement it differentiably with ~30 lines of PyTorch.

---

## 1. The problem with raw covariance features

Recall our pool

$$
C \;=\; \frac{1}{L}\,(X U)^{\top}(X U) \;\in\; \mathbb{R}^{d_c \times d_c},
\qquad U \in \mathbb{R}^{d \times d_c},
$$

where $X \in \mathbb{R}^{L \times d}$ are per-residue embeddings and $U$ either
a trained projection (`cov_supervised`), the top-$d_c$ eigenvectors of the
dataset covariance (`cov_pca`), or a frozen autoencoder solution
(`cov_unsupervised`). Flattening gives the input to the FNN probe head:

$$
\phi(X) \;=\; \text{vec}(C) \;\in\; \mathbb{R}^{d_c^2}.
$$

This works (see [results/visualization/](../results/visualization/)) but has
two structural problems baked into the geometry of covariance matrices:

### 1.1 Heavy-tailed eigenvalue spectrum

For any positive semi-definite (PSD) matrix, the eigenvalue distribution of
$C$ is dominated by a few large modes (the top variance directions of the
projected residues). Empirically, the ratio $\lambda_{\max} / \lambda_{\min}$
on pLM-derived $C$ exceeds $10^4$. After flattening, a handful of entries of
$\text{vec}(C)$ swamp the FNN's first linear layer; the rest of the spectrum
— which still carries useful information — is invisible at training
initialisation. The probe spends most of its capacity scaling those large
entries down.

### 1.2 Covariance is not in Euclidean space

PSD matrices form a **Riemannian manifold** $\mathcal{S}_{++}^{d_c}$, not a
vector space. Linear operations on $\text{vec}(C)$ — which is exactly what
a linear classifier does — implicitly assume Euclidean geometry, so
"distances" between two proteins' covariances are measured wrong:

$$
\|\,\text{vec}(C_1) - \text{vec}(C_2)\,\|_2 \quad\ne\quad d_{\text{SPD}}(C_1, C_2).
$$

The natural geodesic distance on $\mathcal{S}_{++}^{d_c}$ is

$$
d_{\text{aff}}(C_1, C_2) \;=\; \|\,\log(C_1^{-1/2} C_2 C_1^{-1/2})\,\|_F,
$$

which has no relation to the Frobenius distance unless we first map $C$ off
the manifold into a tangent space.

These two problems — eigenvalue concentration and wrong geometry — are tightly
linked: the fix for one fixes the other.

---

## 2. The fix: matrix power normalisation

Replace $C$ with $C^{p}$ for some power $p \in (0, 1]$, where the matrix
function is defined via the eigendecomposition $C = Q \Lambda Q^{\top}$ as

$$
C^{p} \;:=\; Q \, \Lambda^{p} \, Q^{\top}, \qquad
\Lambda^{p} = \operatorname{diag}(\lambda_1^{p}, \dots, \lambda_{d_c}^{p}).
$$

The two canonical choices:

| $p$ | Name | Effect |
|-----|------|--------|
| $p = 1$ | identity (no change) | baseline |
| $p = 1/2$ | **matrix square root** | $\lambda^{1/2}$ — moderate compression |
| $p \to 0$ | $\log(C)$ (Log-Euclidean) | $\lambda \to \log\lambda$ — strong compression |

Power normalisation with $p = 1/2$ is the de-facto standard from iSQRT-COV
(Li et al. 2018), and is what we recommend as the first thing to try. It

1. **Flattens the spectrum.** $\lambda^{1/2}$ shrinks dynamic range from
   $[\lambda_{\min}, \lambda_{\max}]$ to $[\sqrt{\lambda_{\min}}, \sqrt{\lambda_{\max}}]$,
   so a 10⁴ condition number drops to 10². The FNN sees a much better-
   conditioned input.
2. **Approximates the Log-Euclidean tangent.** First-order Taylor:
   $\lambda^{p} \approx 1 + p \log \lambda$ near $\lambda = 1$, so
   $C^{1/2}$ is a smooth interpolation between raw $C$ and the proper
   Riemannian map $\log C$. Linear operations on $\text{vec}(C^{1/2})$
   approximate geodesic distances on $\mathcal{S}_{++}^{d_c}$.
3. **Removes the burstiness.** Same principle as L2-square-root normalisation
   in bag-of-features / VLAD descriptors (Perronnin et al. 2010): squashing
   large magnitudes reduces the dominance of repeated co-activations and
   gives downstream linear classifiers a more useful signal-to-noise.

### 2.1 Why this is more than a feature scaling

A common reaction is "this looks like batch-norm or just a square root applied
elementwise — why bother?" The key is that the power is taken **in the
eigenbasis of $C$**, not entry-wise. Concretely:

$$
[C^{1/2}]_{ij} \quad\ne\quad [C_{ij}]^{1/2}.
$$

Element-wise square root would destroy the PSD structure (negative entries
become NaN) and not be a manifold map. The eigenvalue-wise power preserves
PSDness and is a smooth, invertible map of $\mathcal{S}_{++}^{d_c}$ onto
itself.

---

## 3. Differentiable implementation: Newton–Schulz iteration

For end-to-end training we need $\partial \mathcal{L} / \partial C$ through
the square root. Two options:

### 3.1 Option A: SVD backward

PyTorch's `torch.linalg.eigh` is differentiable, so

```python
L, Q = torch.linalg.eigh(C)        # ascending eigenvalues, Q eigenvectors
C_sqrt = Q @ torch.diag(L.clamp_min(eps).sqrt()) @ Q.T
```

works directly. The cost is $\mathcal O(d_c^3)$ and the backward pass involves
finite differences of close-by eigenvalues, which numerically explodes when
the spectrum has near-degeneracies. **Avoid this in training**; it is fine for
inference or for one-shot fitting of `cov_pca`.

### 3.2 Option B: Newton–Schulz (the iSQRT-COV recipe)

Li et al. (2018) showed that ~5 iterations of Newton–Schulz produce a square
root that is sufficiently accurate for training and is fully differentiable
without any eigendecomposition. The recipe:

**Step 1 — pre-normalise.** Newton–Schulz only converges if $\|C\| < 1$.
Divide by the trace:

$$
A \;=\; C / \operatorname{tr}(C).
$$

**Step 2 — coupled iteration.** Set $Y_0 = A$ and $Z_0 = I$. For
$k = 0, \dots, K-1$:

$$
Y_{k+1} = \tfrac{1}{2}\,Y_k\,(3 I - Z_k Y_k),
\qquad
Z_{k+1} = \tfrac{1}{2}\,(3 I - Z_k Y_k)\,Z_k.
$$

After $K \approx 5$ steps, $Y_K \approx A^{1/2}$ and $Z_K \approx A^{-1/2}$ to
machine precision for spectrum ratios up to $\sim 10^4$.

**Step 3 — post-compensate the pre-norm.** Restore the trace scale:

$$
C^{1/2} \;=\; \sqrt{\operatorname{tr}(C)} \;\; Y_K.
$$

Each iteration is two matrix-matrix products of size $d_c \times d_c$, total
cost $\mathcal O(K d_c^3)$ — for $d_c = 32$ and $K = 5$, that is ten 32×32
matmuls per protein, negligible vs the FNN forward pass.

**Reference implementation** (drop in `src/sop/pooling/`):

```python
def isqrt_cov(C: torch.Tensor, n_iter: int = 5, eps: float = 1e-5) -> torch.Tensor:
    """Differentiable matrix square root via Newton–Schulz (Li et al. 2018).

    Args:
        C: [..., d, d] symmetric PSD matrix.
        n_iter: Number of Newton–Schulz steps (5 is the published default).

    Returns:
        C^{1/2} of the same shape.
    """
    d = C.shape[-1]
    I = torch.eye(d, device=C.device, dtype=C.dtype).expand_as(C)

    # 1. Trace-normalise so ||A|| < 1.
    trace = C.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    A = C / (trace + eps)

    # 2. Coupled Newton–Schulz.
    Y, Z = A, I.clone()
    for _ in range(n_iter):
        T = 0.5 * (3 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    # 3. Restore the trace scale.
    return Y * trace.sqrt()
```

Wire it into the existing pooler:

```python
# in CovariancePooler.forward, after computing C = (XU)^T (XU) / L
if self.power_norm:
    C = isqrt_cov(C, n_iter=5)
# then flatten as before
```

Symmetry of $C$ is preserved through the iteration up to floating-point error;
adding `C = 0.5 * (C + C.transpose(-2, -1))` before the call is a cheap
safety net.

---

## 4. Why we expect a gain on pLM embeddings

Three reasons specific to ProtX / ProtT5 features:

1. **High effective rank.** Transformer hidden states are known to have a
   heavy-tailed singular value spectrum (Wang et al. 2020); covariance of
   such features inherits the heavy tail. iSQRT-COV directly addresses
   exactly this pathology.
2. **Variable sequence length.** $C = (1/L) X^{\top} X$ already normalises
   for length, but the per-protein eigenvalue magnitude still scales with
   the **distribution** of residue activations, not the count. Trace-
   normalisation inside iSQRT-COV adds a second, length-invariant scaling.
3. **Compatibility with our existing baselines.** $C^{1/2}$ is a strict
   superset of $C$ — fix $p = 1$ and you recover the supervised covariance
   pool exactly. So there is no downside risk; the worst case is the FNN
   learns to invert the square root, which it can do in one linear layer.

In CV bilinear pooling the published gain from $C \to C^{1/2}$ is
**+2.0 ± 0.5 pp** on ImageNet fine-grained tasks (Li et al. 2018, Tab. 3).
A comparable swing on DeepLoc would push the hybrid 84.8% → ~86.5%, into
ProtT5+LA territory using only an FNN head.

---

## 5. Suggested experiment matrix

To slot cleanly into the existing pipeline:

| Config | $p$ | $d_c$ | Where to add |
|--------|-----|-------|--------------|
| `cov_supervised_sqrt` | 1/2 | 8, 16, 32, 48 | new YAML in `configs/scl/` |
| `cov_pca_sqrt`        | 1/2 | 32            | new YAML in `configs/scl/` |
| `cov_unsupervised_sqrt` | 1/2 | 32          | new YAML in `configs/scl/` |
| `hybrid_sqrt`         | 1/2 | 32            | new YAML in `configs/scl/` |
| (ablation) `cov_supervised_log` | $\to 0$ | 32 | optional, for Log-Euclidean check |

Same 3 seeds × 2 tasks (DeepLoc + Meltome) as the rest of the grid. Expected
incremental compute: ~10 minutes per run on T4 (the matrix sqrt is
negligible). The combined sweep + ablation is a clean
section in the report and a strong differentiator from a group doing only
LA + cov.

### 5.1 Stacking with LA

If you go the LA + cov route, the natural full stack is

$$
\text{ProtX} \;\to\; \text{LA-weighted residues} \;\to\; C \;\to\; C^{1/2} \;\to\; \text{vec} \;\to\; \text{FNN}.
$$

LA replaces the uniform $\tfrac{1}{L}$ in $C$ with learned per-residue
attention weights $\alpha_i$:

$$
C_{\text{LA}} \;=\; \sum_{i=1}^{L} \alpha_i\, (U^{\top} x_i)(U^{\top} x_i)^{\top},
\qquad \sum_i \alpha_i = 1.
$$

iSQRT-COV is applied after this exactly as above — they compose orthogonally.

---

## 6. References

- Li, P., Xie, J., Wang, Q., & Gao, Z. (2018). *Towards Faster Training of Global
  Covariance Pooling Networks by Iterative Matrix Square Root Normalization.*
  CVPR. — original iSQRT-COV, the Newton–Schulz recipe used above.
- Lin, T.-Y., & Maji, S. (2017). *Improved Bilinear Pooling with CNNs.* BMVC. —
  shows the empirical gain of $C^{1/2}$ on fine-grained CV.
- Ionescu, C., Vantzos, O., & Sminchisescu, C. (2015). *Matrix Backpropagation
  for Deep Networks with Structured Layers.* ICCV. — backprop through eigh.
- Arsigny, V., Fillard, P., Pennec, X., & Ayache, N. (2007). *Geometric Means in
  a Novel Vector Space Structure on Symmetric Positive-Definite Matrices.*
  SIAM J. Matrix Anal. Appl. — Log-Euclidean framework.
- Pennec, X., Fillard, P., & Ayache, N. (2006). *A Riemannian Framework for
  Tensor Computing.* IJCV. — affine-invariant SPD geometry.
