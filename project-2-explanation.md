# Project 2: Second-Order Pooling for Protein Language Models

## 1. Big Picture (What the project is about)
Protein language models (pLMs) such as ProtX output an embedding for every residue in a protein sequence. Most downstream tasks (classification or regression per protein) need a single fixed-size vector per protein. The standard approach is mean pooling, which averages residue embeddings across the sequence. This is simple, fast, and often good, but it discards interactions between features within a residue.

The project tests whether *second-order pooling* (covariance pooling) captures feature co-activation patterns and improves downstream performance compared to mean pooling. The key idea is to compute a covariance-like matrix of the residue embeddings, then flatten it to a vector and feed it to the same probe head.

## 2. Why mean pooling can miss biology
Let the per-residue embedding be $X \in \mathbb{R}^{L \times d}$ where $L$ is sequence length and $d$ is the embedding dimension. Mean pooling is

$$
\mu = \frac{1}{L} X^\top \mathbf{1}_L \in \mathbb{R}^d
$$

Mean pooling computes each feature's average independently. If two features are important only when they appear together at the *same residue*, mean pooling ignores that co-activation. For example:
- Motifs that require two features to appear together in a loop region
- Feature pairs corresponding to disulfide bond patterns

Mean pooling produces the same vector for sequences where the two features appear together versus separately, which can be biologically wrong.

### Order information is lost
Mean pooling is permutation-invariant: it averages across residues and discards their order. Any reordering of residues produces the same pooled vector. So it keeps a "bag of residues" summary, not the sequence order.

## 3. Covariance pooling (Second-order pooling)
Covariance pooling captures how features co-vary across residues. The project uses a *low-rank* projected covariance:

1. Learn two projection matrices $L, R \in \mathbb{R}^{d \times d_c}$.
2. Project embeddings: $X_L = X L$ and $X_R = X R$.
3. Compute a covariance-like matrix:

$$
C = \frac{1}{L} (X_L)^\top (X_R) \in \mathbb{R}^{d_c \times d_c}
$$

Then flatten $C$ into a vector of size $d_c^2$. This vector is fed into the probe head (classifier/regressor). The model still uses a frozen ProtX backbone.

### What does $C$ capture?
Each entry $C_{i,j}$ measures how often projected feature $i$ and projected feature $j$ are active together at the same residues. This is second-order information that mean pooling discards.

### Covariance pooling at the same level as mean pooling
Mean pooling: average across residues to get one number per feature.

Covariance pooling: instead of one number per feature, compute how *pairs* of features co-activate across residues. After projection into a smaller space ($d_c$), you compute the $d_c \times d_c$ matrix $C$ and flatten it. This gives a fixed-size vector that encodes feature co-occurrence patterns, which mean pooling ignores.

## 4. Training regimes for $L$ and $R$
The project compares two approaches:

### 4.1 Supervised training
Train $L$ and $R$ end-to-end with the task head for each task. This is flexible but can overfit and needs to be retrained per task.

### 4.2 Unsupervised training (autoencoder-like)
Train $L$ and $R$ to reconstruct the *full* covariance $X^\top X$ using a low-rank factorization. This is like a linear autoencoder for the covariance structure. Train once, freeze, reuse on all tasks.

The description notes a Frobenius-norm equivalence to avoid explicitly building per-sequence $X^\top X$ matrices, which are large. The key idea: minimize a loss that matches the second-order structure without materializing the full covariance matrix.

## 5. What is an autoencoder in this context?
An autoencoder is a model that compresses data into a lower-dimensional latent space and then reconstructs the original data. Here:
- Input: the full covariance matrix (or an equivalent representation of its information)
- Bottleneck: $d_c$ (the projected dimension)
- Output: reconstructed covariance

Because $L$ and $R$ are linear projections, this is effectively a **linear autoencoder** for covariance structure. It is unsupervised because no task labels are used; only the internal structure of $X$ is reconstructed.

## 6. Could we skip training the autoencoder and use an algorithmic covariance table?
Short answer: you can use an *explicit covariance* or a *fixed projection*, but it will be expensive or less effective. Here is the landscape:

### 6.1 Direct covariance matrix (no training)
Compute $X^\top X / L$ directly and flatten it. This is the purest second-order pooling but has size $d^2$, which is huge for pLMs (e.g., $d=1024 \Rightarrow d^2=1,048,576$). This is usually too big for practical training and storage.

### 6.2 Fixed random projections (no training)
Use random $L$ and $R$ (e.g., Gaussian or orthogonal) and compute $C$. This gives a *sketch* of the covariance (a Johnson-Lindenstrauss style projection). It is cheap, but usually inferior to learned projections. It can be a useful baseline.

### 6.3 Analytical covariance estimators
You could attempt shrinkage covariance estimators or kernelized covariance, but they still face the $d^2$ size issue. These are not drop-in replacements for the learned low-rank covariance, because you still need a compact representation.

### 6.4 Practical answer to the question
Yes, you can replace the autoencoder with a fixed, algorithmic covariance sketch, but it is likely to underperform the learned projection. The project includes an unsupervised option specifically because it is a learnable, task-agnostic way to compress covariance.

## 6.5 Why not keep the full $d \times L$ matrix as the protein embedding?
Keeping the full residue matrix is not a fixed-size embedding because $L$ varies across proteins. It also makes downstream models large and costly because they must re-process long sequences. Pooling exists to produce a compact, fixed-size representation that is cheap to store, compare, and feed into simple heads. If you keep the whole matrix, you are effectively building another sequence model on top.

## 6.6 Standard use cases for per-protein embeddings
- Classification: subcellular localization, enzyme class, membrane vs soluble
- Regression: stability/thermostability, solubility, expression yield
- Similarity search and retrieval: nearest neighbors for functional or structural analogs
- Clustering and visualization: grouping proteins, UMAP/t-SNE plots
- Annotation transfer: assign labels to new proteins based on embedding neighbors
- Screening and prioritization: rank candidates in large protein libraries

## 6.7 CV analogies (how second-order pooling is used elsewhere)
The computer-vision literature uses bilinear or covariance pooling to capture feature co-activations across spatial positions. The mapping to proteins is direct: spatial positions in an image correspond to residues in a sequence.

### Bilinear CNNs (outer product over positions)
- CV pattern: compute outer products of feature vectors at each spatial location and pool across all locations.
- Protein mapping: compute outer products of residue embeddings and average across residues.
- Why it matters: it captures co-activation patterns, not just average activity.
- Reference: Bilinear CNNs for Fine-grained Visual Recognition (arXiv:1504.07889).

### Compact bilinear pooling (reduce d^2)
- CV pattern: the full bilinear feature is huge; compact sketches or projections approximate it with fewer dimensions.
- Protein mapping: use learnable projections L and R into d_c, so covariance is d_c^2 instead of d^2.
- Reference: Compact Bilinear Pooling (arXiv:1511.06062).

### Global covariance pooling with matrix normalization
- CV pattern: treat the pooled covariance as an SPD matrix and apply matrix square-root or power normalization for stability.
- Protein mapping: optional normalization step on C before flattening; can improve training stability.
- Reference: iSQRT-COV / Global Covariance Pooling (arXiv:1712.01034).

### Recent uses outside classical CV
- Pose estimation: covariance pooling as a structured representation for end-to-end pose regression (arXiv:2603.19961).
- Medical imaging: covariance descriptors with SPD-aware networks (arXiv:2511.04190, arXiv:2603.26351).

### Summary for this project
- The protein setup mirrors the CV setup: replace spatial positions with residues.
- The same trade-off applies: second-order info vs. embedding size.
- Compact/low-rank projection is the practical way to make it work.

## 7. Experimental design in this project
The main experiments compare 4 pooling methods:
1. Mean pooling (baseline)
2. Covariance pooling, supervised
3. Covariance pooling, unsupervised (frozen)
4. Hybrid: concatenate mean + covariance

Tasks (at least two):
- Subcellular localization (classification)
- Thermostability (regression)

Additional experimental sweeps:
- Size sweep for $d_c \in \{8, 16, 24, 32, 48\}$ to find the best trade-off
- Layer sweep (last, last 4, middle, early)
- Possibly additional baselines (light attention, pooling alternatives)

## 8. Math intuition for parameter efficiency
Mean pooling gives a $d$-dim vector. Covariance pooling gives $d_c^2$ features, but $d_c$ can be much smaller than $d$. The question is whether $d_c^2$ features can match or beat $d$ features in downstream performance.

The project asks: at what $d_c$ does covariance pooling become *parameter-efficient*, meaning it matches or surpasses mean pooling using fewer or similar parameters in the probe head.

## 9. Critical implementation points
- Proper masking of padded residues is essential for both mean and covariance pooling.
- Caching embeddings from ProtX avoids recomputation and keeps experiments fast.
- All pooling methods must share the same probe head to make comparisons fair.

## 10. Suggested next steps (actionable)
1. Implement and unit-test masked mean pooling and masked covariance pooling side-by-side.
2. Run a small sanity check on a tiny dataset (one task, one seed) to confirm training pipeline.
3. Run the head-to-head benchmark for all 4 pooling methods on the two tasks.
4. Run the $d_c$ size sweep and plot performance vs. embedding size.
5. Run the layer sweep for a subset of layers to test robustness.
6. If time permits, add one additional pooling baseline (light attention).
7. Create the required plots (bar chart, size curve, heatmap) for final presentation.

## 11. How to explain the project in a presentation
- Start with mean pooling and its limitation (independent averaging).
- Introduce second-order pooling as capturing co-activation of features.
- Explain the low-rank projection ($L$ and $R$) to keep size manageable.
- Present the two training regimes (supervised vs. unsupervised), linking the latter to autoencoders.
- Emphasize the experiments that compare performance and efficiency.
- End with the key result you expect: improved performance or similar performance at smaller embedding size.

## 12. Short glossary
- pLM: protein language model
- Residue embedding: vector for each amino acid position
- Mean pooling: average across residues
- Covariance pooling: second-order statistics across residues
- Probe head: the classifier/regressor on top of pooled embeddings
- Bottleneck: reduced dimension $d_c$

---

If you want, I can also draft a slide outline or add a more formal math derivation section.