"""Vision model: ResNet-18 fine-tuning for binary skin-lesion classification.

All sampling methods act through a per-epoch WeightedRandomSampler (weights
normalized from the method's sample weights), so every method sees identical
architecture, initialization (ImageNet-pretrained ResNet-18), augmentation,
optimizer, epochs, and batch size. Threshold is selected on validation BAcc.

Checkpointing: each run saves state to results/vision_ckpt/<run>.pt and is
resumable. GPU: single consumer GPU <=16 GB (tested on RTX 4070 Ti SUPER).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import ResNet18_Weights, resnet18

CKPT_DIR = Path(__file__).resolve().parents[3] / "results" / "vision_ckpt"


class MemmapImageDataset(Dataset):
    """uint8 memmap (N,H,W,3) with labels + optional sample index mapping.

    ``index`` maps dataset positions to memmap rows. When a split (train/val/
    test subset) is passed as ``y``, the mapping is REQUIRED to read the right
    images; without it the loader would silently read memmap rows by subset
    position. Flip augmentation parity is driven by the memmap row, so a given
    image is always augmented identically regardless of subset membership.
    """

    def __init__(self, images_path: str | Path, y: np.ndarray, train: bool = True,
                 index: np.ndarray | None = None):
        self.images = np.load(images_path, mmap_mode="r")
        self.y = np.asarray(y, dtype=np.float32)
        self.index = np.arange(len(self.y)) if index is None else np.asarray(index)
        assert len(self.index) == len(self.y), "index/y length mismatch"
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row = int(self.index[idx])
        img = torch.from_numpy(np.ascontiguousarray(self.images[row])).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        if self.train:
            # deterministic-flip augmentation driven by memmap row parity
            if (row % 2 == 0):
                img = torch.flip(img, dims=[2])
        return img, self.y[idx], int(idx)


def build_model(seed: int, device: str) -> nn.Module:
    torch.manual_seed(seed)
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m.to(device)


def _weights_for_epoch(method_weights: np.ndarray) -> np.ndarray:
    w = np.asarray(method_weights, dtype=np.float64)
    return w / w.sum()


def train_one(
    images_path: str | Path,
    y_tr: np.ndarray,
    idx_tr: np.ndarray,
    y_val: np.ndarray,
    idx_val: np.ndarray,
    method_weights: np.ndarray | None,
    *,
    seed: int = 0,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str | None = None,
    run_id: str = "",
    dro_bins: np.ndarray | None = None,
    dro_eta: float = 0.5,
    tag: str | None = None,
) -> dict:
    """Train with a weighted sampler; returns val scores/threshold info.

    ``method_weights`` are full-length training weights aligned to ``idx_tr``.
    Iterative proxy-DRO: reweight bins by val loss each epoch (simple
    exponential-weights variant adapted to vision; deterministic given seed).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed % 2**31)

    ds = MemmapImageDataset(images_path, y_tr, train=True, index=idx_tr)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(_weights_for_epoch(method_weights)),
        num_samples=len(y_tr),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=False)
    val_ds = MemmapImageDataset(images_path, y_val, train=False, index=idx_val)
    # batch size pinned to predict_scores' 512 so thresholds are exactly
    # reproducible from checkpoints (float accumulation is batch-size
    # dependent; see results/audit/protocol_verification.json)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    cur_w = np.asarray(method_weights, dtype=float).copy()
    bins = dro_bins
    bce = nn.BCEWithLogitsLoss(reduction="none")

    ckpt_name = f"{run_id}.pt" if tag in (None, "main") else f"{run_id}@{tag}.pt"
    ckpt_path = CKPT_DIR / ckpt_name
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    state = {"epoch": 0, "weights": cur_w.tolist() if bins is not None else None}
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        if sd.get("epochs_requested") != epochs:
            # checkpoint from a different protocol (e.g. a sensitivity arm
            # overwrote a main-grid slot): discard and train fresh
            ckpt_path.unlink()
            start_epoch = 0
        else:
            model.load_state_dict(sd["model"])
            opt.load_state_dict(sd["opt"])
            sched.load_state_dict(sd["sched"])
            start_epoch = sd["epoch"] + 1
            if bins is not None and "weights" in sd:
                cur_w = np.asarray(sd["weights"], dtype=float)
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        if bins is not None and epoch > 0:
            # recompute sampler weights from last epoch's training losses
            model.eval()
            losses = np.zeros(len(y_tr))
            with torch.no_grad():
                for x, yy, ii in loader:
                    out = model(x.to(device)).squeeze(1).cpu()
                    l = bce(out, yy)
                    losses[ii.numpy()] = l.numpy()
            model.train()
            from ..sampling.methods import dro_update_weights

            cur_w = dro_update_weights(losses, bins, cur_w, dro_eta)
            sampler.weights = torch.as_tensor(_weights_for_epoch(cur_w))
        for x, yy, _ in loader:
            x, yy = x.to(device, non_blocking=True), yy.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x).squeeze(1)
            loss = loss_fn(out, yy)
            loss.backward()
            opt.step()
        sched.step()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch,
                    "epochs_requested": epochs,
                    "weights": cur_w.tolist() if bins is not None else None}, ckpt_path)

    # validation scores
    model.eval()
    scores, labels, idxs = [], [], []
    with torch.no_grad():
        for x, yy, ii in val_loader:
            s = torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy()
            scores.append(s)
            labels.append(yy.numpy())
            idxs.append(ii.numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    idxs = np.concatenate(idxs)
    return {"scores": scores, "labels": labels, "idx": idxs, "epochs_done": epochs,
            "model": model}


def predict_scores(model: nn.Module, images_path, y, index=None, batch_size=512, device=None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = MemmapImageDataset(images_path, y, train=False, index=index)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    out = []
    with torch.no_grad():
        for x, _, _ in loader:
            s = torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy()
            out.append(s)
    return np.concatenate(out)
