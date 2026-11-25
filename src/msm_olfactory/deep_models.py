"""The three neural classifiers compared in the source (Steps 16-19), with data loading, training and timing.

* :class:`TicOnlyCNN`: 3 conv layers on the 1 x 4 x W TIC image. The class name follows the source, but the image
  contains MSM-regime and ADF/KPSS rows, so this is not an intensity-only model ("TIC-image CNN" in reports).
* :class:`MsmOnlyMLP`: a 64-32 MLP on the 30-dimensional ComBat-harmonized feature vector (which includes age and
  PSA).
* :class:`MultiModalResNet`: a ResNet18 (random initialisation, 1 channel) on the image upsampled to 224 x 224, an
  MLP on the feature vector, and a learned scalar gate g: ``fused = g * image_embedding + (1 - g) * tabular_embedding``.

Training follows the source: Adam (lr 1e-3), cross-entropy, batch 8, 20 epochs. The reported score is the best
validation F1 over the epochs, on the same validation split (no separate test set).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - conventional PyTorch alias
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from msm_olfactory.config import CNN_NAME, MLP_NAME, MULTIMODAL_NAME, N_IMAGE_ROWS, Settings
from msm_olfactory.evaluation import always_cancer_f1
from msm_olfactory.images import load_tic_image

logger = logging.getLogger(__name__)

RESNET_INPUT_SIZE = (224, 224)  # ResNet18's native input resolution
EMBEDDING_DIM = 128  # shared size of the image and tabular embeddings before gated fusion
N_CLASSES = 2


class TicMsmDataset(Dataset):
    """Pairs each sample's TIC image with its feature vector and label.

    Parameters
    ----------
    records
        One dict per sample: ``{"sample_id", "path" (PNG), "row" (index into X_tab), "label" (0/1)}``.
    X_tab
        Feature matrix, shape ``(n_samples, n_features)``.
    """

    def __init__(self, records: list[dict], X_tab: np.ndarray):
        """Store the records and the float32 feature matrix."""
        self.records = records
        self.X_tab = np.asarray(X_tab, dtype=np.float32)

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(image of shape (1, H, W), feature vector, label)``."""
        rec = self.records[idx]
        img = torch.from_numpy(load_tic_image(rec["path"])[None, :, :])
        tab = torch.from_numpy(self.X_tab[rec["row"]])
        return img, tab, torch.tensor(int(rec["label"]), dtype=torch.long)


class TicOnlyCNN(nn.Module):
    """Small CNN on the 4-row TIC image; pooling acts along time only, so the 4 rows are preserved."""

    def __init__(self, img_height: int = N_IMAGE_ROWS, img_width: int = 256, num_classes: int = N_CLASSES):
        """Build the layers; the flattened feature size is inferred from a dummy image of the given size."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = (1, 2)
        with torch.no_grad():
            feat_dim = self._features(torch.zeros(1, 1, img_height, img_width)).shape[1]
        self.fc1 = nn.Linear(feat_dim, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=self.pool, stride=self.pool)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=self.pool, stride=self.pool)
        return torch.flatten(F.relu(self.conv3(x)), 1)

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor | None = None) -> torch.Tensor:
        """Class logits from the image (the feature vector is ignored)."""
        return self.fc_out(F.relu(self.fc1(self._features(x_img))))


class MsmOnlyMLP(nn.Module):
    """Two-hidden-layer MLP (64, 32) on the feature vector."""

    def __init__(self, tab_in_dim: int, num_classes: int = N_CLASSES):
        """Build the layers for ``tab_in_dim`` input features."""
        super().__init__()
        self.fc1 = nn.Linear(tab_in_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        """Class logits from the feature vector (the image is ignored)."""
        return self.fc_out(F.relu(self.fc2(F.relu(self.fc1(x_tab)))))


class MultiModalResNet(nn.Module):
    """ResNet18 image branch + MLP feature branch, fused by a learned scalar gate."""

    def __init__(self, tab_in_dim: int, num_classes: int = N_CLASSES):
        """Build both branches, the gate and the classifier head."""
        super().__init__()
        self.upsample = nn.Upsample(size=RESNET_INPUT_SIZE, mode="bilinear")
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)  # 1-channel input
        self.resnet.fc = nn.Identity()  # expose the 512-D embedding
        self.img_proj = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, EMBEDDING_DIM))
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_in_dim, EMBEDDING_DIM),
            nn.ReLU(),
            nn.BatchNorm1d(EMBEDDING_DIM),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            nn.ReLU(),
            nn.BatchNorm1d(EMBEDDING_DIM),
        )
        self.gate = nn.Sequential(nn.Linear(2 * EMBEDDING_DIM, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(EMBEDDING_DIM, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, img: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        """Class logits from the gated combination of the image and feature embeddings."""
        img_feat = self.img_proj(self.resnet(self.upsample(img)))
        tab_feat = self.tab_proj(tab)
        g = self.gate(torch.cat([img_feat, tab_feat], dim=1))  # (B, 1), in (0, 1)
        return self.classifier(g * img_feat + (1 - g) * tab_feat)


def build_models(tab_in_dim: int, image_width: int) -> dict[str, nn.Module]:
    """Instantiate the three compared models, in the source's order (this order consumes the PyTorch RNG)."""
    return {
        CNN_NAME: TicOnlyCNN(img_width=image_width),
        MLP_NAME: MsmOnlyMLP(tab_in_dim),
        MULTIMODAL_NAME: MultiModalResNet(tab_in_dim),
    }


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_loaders(
    train_records: list[dict],
    val_records: list[dict],
    X_tab: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Training loader (shuffled with a seeded generator) and validation loader (fixed order)."""
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(TicMsmDataset(train_records, X_tab), batch_size=batch_size, shuffle=True, generator=generator)
    val = DataLoader(TicMsmDataset(val_records, X_tab), batch_size=batch_size, shuffle=False)
    return train, val


def _run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    """One pass over ``loader``: trains if ``optimizer`` is given, otherwise evaluates. Returns (mean loss, F1)."""
    training = optimizer is not None
    model.train(training)
    total, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for img, tab, y in loader:
            img, tab, y = img.to(device), tab.to(device), y.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(img, tab)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()
            total += loss.item() * y.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return total / len(loader.dataset), float(f1_score(labels, preds, zero_division=0))


@dataclass
class TrainingResult:
    """Per-epoch history and best validation F1 of one model."""

    name: str
    best_val_f1: float
    best_epoch: int
    history: list[dict] = field(default_factory=list)

    @property
    def final_val_f1(self) -> float:
        """Validation F1 at the last epoch (no selection on the validation split)."""
        return self.history[-1]["val_f1"]


def train_model(
    name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, device
) -> TrainingResult:
    """Train with Adam and cross-entropy, recording train/validation loss and F1 for every epoch.

    ``best_val_f1`` is the maximum over epochs, so the epoch is selected on the validation split itself (the source
    protocol). It is an optimistic estimate; :attr:`TrainingResult.final_val_f1` is the unselected value.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    result = TrainingResult(name=name, best_val_f1=0.0, best_epoch=0)
    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_f1 = _run_epoch(model, val_loader, criterion, device)
        result.history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_f1": train_f1, "val_loss": val_loss, "val_f1": val_f1}
        )
        if val_f1 > result.best_val_f1:
            result.best_val_f1, result.best_epoch = val_f1, epoch
    return result


@dataclass
class ModelComparison:
    """Outcome of training the three models on one split with one PyTorch seed."""

    runs: dict[str, TrainingResult]
    models: dict[str, nn.Module]
    train_records: list[dict]
    val_records: list[dict]
    val_loader: DataLoader
    always_cancer_val_f1: float


def train_and_compare_models(
    records: list[dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X_tab: np.ndarray,
    settings: Settings,
    torch_seed: int,
) -> ModelComparison:
    """Train the TIC-image CNN, the MSM-feature MLP and the multimodal model on the same split (Steps 16-18).

    Parameters
    ----------
    records
        One record per labelled sample (see :class:`TicMsmDataset`); ``row`` indexes ``X_tab``.
    train_idx, val_idx
        Positions in ``records`` from :func:`msm_olfactory.evaluation.stratified_holdout_indices`.
    X_tab
        Harmonized feature matrix.
    settings
        Batch size, epochs, learning rate and image width.
    torch_seed
        Seeds weight initialisation and training-batch order (the source was unseeded).

    Raises
    ------
    ValueError
        If ``X_tab`` contains NaN or does not have one row per record.
    """
    if np.isnan(X_tab).any():
        raise ValueError("Feature matrix contains NaN; the deep models cannot be trained on it.")
    if X_tab.shape[0] != len(records):
        raise ValueError(f"Feature matrix has {X_tab.shape[0]} rows for {len(records)} records.")
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    torch.manual_seed(torch_seed)
    train_loader, val_loader = make_loaders(train_records, val_records, X_tab, settings.batch_size, torch_seed)
    models = build_models(X_tab.shape[1], settings.image_width)
    device = torch.device("cpu")
    logger.info(
        "Training 3 models on %d train / %d validation samples (%d epochs, CPU, torch seed %d)",
        len(train_records),
        len(val_records),
        settings.epochs,
        torch_seed,
    )
    runs = {
        name: train_model(name, model, train_loader, val_loader, settings.epochs, settings.learning_rate, device)
        for name, model in models.items()
    }
    for name, run in runs.items():
        logger.info("  %-27s best validation F1 %.3f (epoch %d)", name, run.best_val_f1, run.best_epoch)
    return ModelComparison(
        runs=runs,
        models=models,
        train_records=train_records,
        val_records=val_records,
        val_loader=val_loader,
        always_cancer_val_f1=always_cancer_f1([r["label"] for r in val_records]),
    )


def time_forward_pass_on_loader(model: nn.Module, loader: DataLoader, n_batches: int = 20) -> float:
    """Mean wall-clock seconds per batch for a forward pass over ``loader`` (source Step 19 protocol).

    Feature extraction is not included. See :mod:`msm_olfactory.benchmarking` for the more stable single-thread
    protocol and the end-to-end costs.

    Raises
    ------
    ValueError
        If the loader yields no batches.
    """
    model.eval()
    times = []
    with torch.no_grad():
        for i, (img, tab, _) in enumerate(loader):
            if i >= n_batches:
                break
            start = time.perf_counter()
            model(img, tab)
            times.append(time.perf_counter() - start)
    if not times:
        raise ValueError("Loader produced no batches")
    return sum(times) / len(times)
