from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset


SEED = 632901
EXPLANATION_SEED = 632902
MEDMNIST_URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
MEDMNIST_MD5 = "28209eda62fecd6e6a2d98b1501bb15f"
OFFICIAL_COMMIT = "70b6b3a7ad7afddff1df2a3b735235830fbdb142"
OFFICIAL_REFERENCE = {"auroc": 0.944, "accuracy": 0.854}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Refusing to save an empty CSV")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(path)
    return path


def obtain_source_data(root: Path) -> Path:
    path = root / "data" / "pneumoniamnist.npz"
    download(MEDMNIST_URL, path)
    actual = md5(path)
    if actual != MEDMNIST_MD5:
        raise RuntimeError(f"PneumoniaMNIST MD5 mismatch: {actual}")
    return path


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        arrays = {key: z[key].copy() for key in z.files}
    expected = {"train": 4708, "val": 524, "test": 624}
    for split, n in expected.items():
        images = arrays[f"{split}_images"]
        labels = arrays[f"{split}_labels"].reshape(-1).astype(np.int64)
        if len(images) != n or len(labels) != n or images.shape[1:3] != (28, 28):
            raise AssertionError((split, images.shape, labels.shape))
        if set(np.unique(labels)) != {0, 1}:
            raise AssertionError((split, np.unique(labels)))
        arrays[f"{split}_labels"] = labels
    return arrays


def normalized_tensor(images: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(images).float()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    elif x.ndim == 4 and x.shape[-1] == 1:
        x = x.permute(0, 3, 1, 2)
    if x.ndim != 4 or x.shape[1] != 1:
        raise AssertionError(x.shape)
    return (x / 255.0 - 0.5) / 0.5


def loader(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(normalized_tensor(images), torch.from_numpy(labels).long())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=(2, 2, 2, 2), in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(in_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for value in strides:
            layers.append(block(self.in_planes, planes, value))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def resnet18() -> ResNet:
    return ResNet(BasicBlock, (2, 2, 2, 2), in_channels=1, num_classes=2)


def predict(model: nn.Module, data_loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            logits.append(model(x.to(device)).cpu())
            labels.append(y)
    logits_np = torch.cat(logits).numpy()
    labels_np = torch.cat(labels).numpy()
    probabilities = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()[:, 1]
    return logits_np, probabilities, labels_np


def threshold_youden(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    index = np.argmax((tpr - fpr)[finite])
    return float(thresholds[finite][index])


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "normal": int((labels == 0).sum()),
        "pneumonia": int((labels == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "brier": float(brier_score_loss(labels, scores)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = SEED
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    milestones: tuple[int, int] = (50, 75)
    gamma: float = 0.1
    selection_metric: str = "validation_auroc"
    augmentation: str = "none"


def prepare(root: Path, source_path: Path, arrays: dict[str, np.ndarray]) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / "runs" / run_id
    out.mkdir(parents=True)
    config = TrainingConfig()
    protocol = {
        "status": "protocol_frozen_before_training",
        "created_utc": utcnow(),
        "paper": "MedMNIST v2, Scientific Data 2023",
        "official_experiments_commit": OFFICIAL_COMMIT,
        "dataset": "PneumoniaMNIST 28",
        "source_sha256": sha256(source_path),
        "source_md5": md5(source_path),
        "splits": {s: int(len(arrays[f"{s}_labels"])) for s in ("train", "val", "test")},
        "config": asdict(config),
        "test_policy": "test is not evaluated per epoch; it is used once after checkpoint selection",
        "explanation_policy": "validation explanations are computed after AUROC checkpoint selection and do not alter checkpoint selection",
        "official_reference": OFFICIAL_REFERENCE,
        "stop_boundary": "stop after external E_diag zero-shot diagnosis; do not train improvement",
    }
    save_json(out / "plan_before_training.json", protocol)
    return out


def train(root: Path, out: Path, arrays: dict[str, np.ndarray]) -> dict:
    config = TrainingConfig()
    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(config.milestones), gamma=config.gamma)
    criterion = nn.CrossEntropyLoss()
    train_loader = loader(arrays["train_images"], arrays["train_labels"], config.batch_size, True, config.seed)
    val_loader = loader(arrays["val_images"], arrays["val_labels"], config.batch_size, False, config.seed)
    rows, best_auc, best_epoch = [], -np.inf, None
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss, count, correct = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss) * len(y)
            count += len(y)
            correct += int((logits.argmax(1) == y).sum())
        val_logits, val_scores, val_labels = predict(model, val_loader, device)
        val_metrics = metrics(val_labels, val_scores, 0.5)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": running_loss / count,
            "train_accuracy": correct / count,
            "val_auroc": val_metrics["auroc"],
            "val_accuracy": val_metrics["accuracy"],
            "elapsed_seconds": time.time() - started,
        }
        rows.append(row)
        if val_metrics["auroc"] > best_auc:
            best_auc, best_epoch = val_metrics["auroc"], epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_auroc": best_auc, "config": asdict(config)}, out / "best_weights.pt")
            np.savez_compressed(out / "best_validation_predictions.npz", logits=val_logits, scores=val_scores, labels=val_labels)
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(f"epoch={epoch:03d} loss={row['train_loss']:.4f} val_auc={row['val_auroc']:.4f} val_acc={row['val_accuracy']:.4f}", flush=True)
    torch.save({"model_state": model.state_dict(), "epoch": config.epochs, "config": asdict(config)}, out / "last_weights.pt")
    save_csv(out / "learning_curve.csv", rows)
    selected = torch.load(out / "best_weights.pt", map_location="cpu", weights_only=False)
    validation = np.load(out / "best_validation_predictions.npz")
    source_threshold = threshold_youden(validation["labels"], validation["scores"])
    summary = {
        "status": "baseline_training_completed_checkpoint_frozen",
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_validation_auroc": float(best_auc),
        "source_youden_threshold": source_threshold,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "training_seconds": time.time() - started,
        "best_weights_sha256": sha256(out / "best_weights.pt"),
        "last_weights_sha256": sha256(out / "last_weights.pt"),
    }
    save_json(out / "training_summary.json", summary)
    save_json(root / "latest_run.json", {"run_id": out.name, "relative_path": f"runs/{out.name}", "status": summary["status"], "best_weights_sha256": summary["best_weights_sha256"]})
    return summary


def load_selected_model(out: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(out / "best_weights.pt", map_location=device, weights_only=False)
    model = resnet18().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def gradcam(model: nn.Module, images: torch.Tensor, device: torch.device, target_class: int = 1) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    storage = {}

    def forward_hook(_module, _inputs, output):
        output.retain_grad()
        storage["activation"] = output

    handle = model.layer4.register_forward_hook(forward_hook)
    try:
        x = images.to(device).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[:, target_class].sum().backward()
        activation = storage["activation"]
        weights = activation.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        maxima = cam.flatten(1).amax(1).view(-1, 1, 1)
        cam = torch.where(maxima > 1e-12, cam / maxima, torch.zeros_like(cam))
        return cam.detach().cpu().numpy(), torch.softmax(logits.detach(), 1)[:, 1].cpu().numpy()
    finally:
        handle.remove()


def stratified_indices(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = []
    for label in (0, 1):
        idx = np.flatnonzero(labels == label)
        n = min(len(idx), maximum // 2)
        groups.append(np.sort(rng.choice(idx, n, replace=False)))
    return np.sort(np.concatenate(groups))


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide((a * b).sum(1), denominator, out=np.zeros(len(a)), where=denominator > 1e-12)


def adjust_brightness(x: torch.Tensor, factor: float) -> torch.Tensor:
    raw = (x * 0.5 + 0.5).clamp(0, 1)
    return ((raw * factor).clamp(0, 1) - 0.5) / 0.5


def adjust_contrast(x: torch.Tensor, factor: float) -> torch.Tensor:
    raw = (x * 0.5 + 0.5).clamp(0, 1)
    mean = raw.mean(dim=(2, 3), keepdim=True)
    return ((mean + factor * (raw - mean)).clamp(0, 1) - 0.5) / 0.5


def border_occlusion(x: torch.Tensor, pixels: int = 2) -> torch.Tensor:
    y = x.clone()
    y[:, :, :pixels, :] = 0
    y[:, :, -pixels:, :] = 0
    y[:, :, :, :pixels] = 0
    y[:, :, :, -pixels:] = 0
    return y


def source_explainability(model: nn.Module, arrays: dict[str, np.ndarray], split: str, out: Path, device: torch.device, maximum: int = 200) -> dict:
    labels = arrays[f"{split}_labels"]
    indices = stratified_indices(labels, maximum, EXPLANATION_SEED + (0 if split == "val" else 1))
    x = normalized_tensor(arrays[f"{split}_images"][indices])
    cams, clean_scores = gradcam(model, x, device)
    bright_cams, bright_scores = gradcam(model, adjust_brightness(x, 1.10), device)
    contrast_cams, contrast_scores = gradcam(model, adjust_contrast(x, 1.10), device)
    with torch.no_grad():
        edge_scores = torch.softmax(model(border_occlusion(x).to(device)), 1)[:, 1].cpu().numpy()
    zero_rate = float((cams.reshape(len(cams), -1).max(1) <= 1e-12).mean())
    result = {
        "split": split,
        "n": int(len(indices)),
        "indices": indices.tolist(),
        "target_class": "pneumonia logit",
        "gradcam_layer": "layer4",
        "zero_cam_rate": zero_rate,
        "brightness_1.10_cam_cosine_mean": float(cosine_rows(cams, bright_cams).mean()),
        "contrast_1.10_cam_cosine_mean": float(cosine_rows(cams, contrast_cams).mean()),
        "brightness_probability_abs_change_mean": float(np.abs(bright_scores - clean_scores).mean()),
        "contrast_probability_abs_change_mean": float(np.abs(contrast_scores - clean_scores).mean()),
        "border_occlusion_probability_abs_change_mean": float(np.abs(edge_scores - clean_scores).mean()),
        "interpretation": "source has no lesion boxes; these are explanation stability and region-dependence proxies, not lesion-localization accuracy",
    }
    np.savez_compressed(out / f"source_{split}_explanations.npz", indices=indices, labels=labels[indices], cams=cams, scores=clean_scores, bright_cams=bright_cams, contrast_cams=contrast_cams, border_scores=edge_scores)
    save_json(out / f"source_{split}_explainability.json", result)
    return result


def plot_learning(out: Path) -> None:
    rows = list(csv.DictReader((out / "learning_curve.csv").open(encoding="utf-8-sig")))
    epochs = np.array([int(r["epoch"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(r["train_loss"]) for r in rows])
    axes[0].set(title="Training loss", xlabel="Epoch")
    axes[1].plot(epochs, [float(r["val_auroc"]) for r in rows], label="AUROC")
    axes[1].plot(epochs, [float(r["val_accuracy"]) for r in rows], label="Accuracy")
    axes[1].axhline(OFFICIAL_REFERENCE["auroc"], color="gray", linestyle="--", linewidth=1, label="Official AUROC")
    axes[1].set(title="Validation metrics", xlabel="Epoch")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=180)
    plt.close(fig)


def evaluate_frozen(root: Path, out: Path, arrays: dict[str, np.ndarray]) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_selected_model(out, device)
    training = json.loads((out / "training_summary.json").read_text(encoding="utf-8"))
    test_loader = loader(arrays["test_images"], arrays["test_labels"], 256, False, SEED)
    logits, scores, labels = predict(model, test_loader, device)
    official = metrics(labels, scores, 0.5)
    source_threshold = metrics(labels, scores, training["source_youden_threshold"])
    np.savez_compressed(out / "internal_test_predictions.npz", logits=logits, scores=scores, labels=labels)
    validation_xai = source_explainability(model, arrays, "val", out, device)
    test_xai = source_explainability(model, arrays, "test", out, device)
    result = {
        "status": "internal_test_completed_baseline_frozen",
        "evaluated_utc": utcnow(),
        "official_argmax_metrics": official,
        "fixed_source_youden_metrics": source_threshold,
        "official_reference": OFFICIAL_REFERENCE,
        "difference_from_official": {"auroc": official["auroc"] - OFFICIAL_REFERENCE["auroc"], "accuracy": official["accuracy"] - OFFICIAL_REFERENCE["accuracy"]},
        "validation_explainability": validation_xai,
        "test_explainability": test_xai,
        "test_predictions_sha256": sha256(out / "internal_test_predictions.npz"),
    }
    save_json(out / "internal_evaluation.json", result)
    plot_learning(out)
    latest = json.loads((root / "latest_run.json").read_text(encoding="utf-8"))
    latest.update(status=result["status"], internal_evaluation_sha256=sha256(out / "internal_evaluation.json"))
    save_json(root / "latest_run.json", latest)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def environment_record() -> dict:
    return {
        "created_utc": utcnow(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
    }


def smoke_test() -> dict:
    seed_all(SEED)
    model = resnet18()
    x = torch.zeros(2, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (2, 2)
    assert sum(p.numel() for p in model.parameters()) == 11168706
    return {"logits_shape": list(logits.shape), "parameter_count": sum(p.numel() for p in model.parameters())}


if __name__ == "__main__":
    project_root = Path(os.environ.get("MD6329_PROJECT_ROOT", Path.cwd()))
    root = project_root / "md6329_step10_resnet_reproduction"
    root.mkdir(parents=True, exist_ok=True)
    source = obtain_source_data(root)
    data = load_arrays(source)
    output = prepare(root, source, data)
    save_json(output / "environment.json", environment_record())
    print(smoke_test())
    train(root, output, data)
    evaluate_frozen(root, output, data)
