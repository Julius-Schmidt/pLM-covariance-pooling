"""Smoke tests for the generic probe training loop.

We don't aim for realistic accuracy — just that the loop runs end-to-end on
both classification and regression and that loss actually decreases on a
deliberately easy synthetic task.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

from sop.pooling.covariance import CovariancePooler
from sop.pooling.mean import MeanPooler
from sop.probes.fnn import ProbeFNN
from sop.probes.model import PoolingProbeModel
from sop.probes.train_loop import train_probe


def _make_loader(B: int, L: int, d: int, n_classes: int, task: str) -> DataLoader:
    torch.manual_seed(0)
    X = torch.randn(B, L, d)
    mask = torch.ones(B, L, dtype=torch.bool)
    if task == "classification":
        # Easy task: pick the class via the sign of the first feature's mean.
        means = X[:, :, 0].mean(dim=1)
        y = (means > 0).long()
    else:
        y = X[:, :, 0].mean(dim=1)

    ds = TensorDataset(X, mask, y)
    return DataLoader(ds, batch_size=8, shuffle=True)


def test_classification_loop_decreases_loss():
    d = 8
    loader = _make_loader(B=32, L=10, d=d, n_classes=2, task="classification")
    pooler = MeanPooler(d)
    probe = ProbeFNN(d, out_dim=2, hidden_dim=16, dropout=0.0)
    model = PoolingProbeModel(pooler, probe)

    result = train_probe(
        model, loader, val_loader=loader, task="classification",
        epochs=10, lr=1e-2, log=lambda _: None,
    )
    first = result["history"][0]["train_loss"]
    last = result["history"][-1]["train_loss"]
    assert last < first, f"Train loss did not decrease: {first:.4f} → {last:.4f}"


def test_regression_loop_decreases_loss():
    d = 8
    loader = _make_loader(B=32, L=10, d=d, n_classes=1, task="regression")
    pooler = MeanPooler(d)
    probe = ProbeFNN(d, out_dim=1, hidden_dim=16, dropout=0.0)
    model = PoolingProbeModel(pooler, probe)

    result = train_probe(
        model, loader, val_loader=loader, task="regression",
        epochs=10, lr=1e-2, log=lambda _: None,
    )
    first = result["history"][0]["train_loss"]
    last = result["history"][-1]["train_loss"]
    assert last < first


def test_supervised_covariance_pooler_grads_flow():
    """When a CovariancePooler is the front-end, its weights must update."""
    d, dc = 8, 4
    loader = _make_loader(B=16, L=6, d=d, n_classes=2, task="classification")
    pooler = CovariancePooler(d, dc)
    probe = ProbeFNN(dc * dc, out_dim=2, hidden_dim=16, dropout=0.0)
    model = PoolingProbeModel(pooler, probe)

    before = pooler.proj_l.weight.detach().clone()
    train_probe(
        model, loader, val_loader=None, task="classification",
        epochs=3, lr=1e-2, log=lambda _: None,
    )
    after = pooler.proj_l.weight.detach()
    assert not torch.allclose(before, after), "Supervised covariance projection did not update"


def test_frozen_covariance_pooler_does_not_update():
    """A frozen CovariancePooler stays put even when training the probe."""
    d, dc = 8, 4
    loader = _make_loader(B=16, L=6, d=d, n_classes=2, task="classification")
    pooler = CovariancePooler(d, dc).freeze()
    probe = ProbeFNN(dc * dc, out_dim=2, hidden_dim=16, dropout=0.0)
    model = PoolingProbeModel(pooler, probe)

    before = pooler.proj_l.weight.detach().clone()
    train_probe(
        model, loader, val_loader=None, task="classification",
        epochs=3, lr=1e-2, log=lambda _: None,
    )
    after = pooler.proj_l.weight.detach()
    assert torch.allclose(before, after), "Frozen pooler weights changed during training"
