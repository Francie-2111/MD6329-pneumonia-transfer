from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


SEED = 632901
DATA_URL = "https://zenodo.org/records/10519652/files/pneumoniamnist_64.npz?download=1"
DATA_MD5 = "8f4eceb4ccffa70c672198ea285246c6"
EXPECTED_SPLITS = {"train": 4708, "val": 524, "test": 624}
B28_VALIDATION_REFERENCE = {"accuracy": 0.9732824427480916, "auroc": 0.996134437779682}
B28_TEST_REFERENCE = {"accuracy": 0.8621794871794872, "auroc": 0.9444115713346481}
CUTMIX_TEST_REFERENCE = {"accuracy": 0.8701923076923077, "auroc": 0.950597194827964}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to save an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
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
    torch.use_deterministic_algorithms(True, warn_only=True)


def obtain_data(root: Path) -> Path:
    path = root / "data" / "pneumoniamnist_64.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".npz.part")
        urllib.request.urlretrieve(DATA_URL, temporary)
        temporary.replace(path)
    actual = md5(path)
    if actual != DATA_MD5:
        raise RuntimeError(f"Dataset MD5 mismatch: {actual} != {DATA_MD5}")
    return path


def load_arrays(path: Path, include_test: bool = False) -> dict[str, np.ndarray]:
    allowed = ("train", "val", "test") if include_test else ("train", "val")
    arrays: dict[str, np.ndarray] = {}
    with np.load(path) as archive:
        for split in allowed:
            images = archive[f"{split}_images"].copy()
            labels = archive[f"{split}_labels"].reshape(-1).astype(np.int64)
            if len(images) != EXPECTED_SPLITS[split] or images.shape[1:3] != (64, 64):
                raise AssertionError((split, images.shape, labels.shape))
            if set(np.unique(labels)) != {0, 1}:
                raise AssertionError((split, np.unique(labels)))
            arrays[f"{split}_images"] = images
            arrays[f"{split}_labels"] = labels
    return arrays


def normalized_tensor(images: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(images).float()
    if value.ndim == 3:
        value = value.unsqueeze(1)
    elif value.ndim == 4 and value.shape[-1] == 1:
        value = value.permute(0, 3, 1, 2)
    if value.ndim != 4 or value.shape[1] != 1:
        raise AssertionError(value.shape)
    return value.div(255.0).sub(0.5).div(0.5)


def make_loader(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(normalized_tensor(images), torch.from_numpy(labels).long())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available(), generator=generator if shuffle else None)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


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
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.bn1(self.conv1(x)))
        value = self.bn2(self.conv2(value)) + self.shortcut(x)
        return F.relu(value)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(1, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._layer(64, 2, 1)
        self.layer2 = self._layer(128, 2, 2)
        self.layer3 = self._layer(256, 2, 2)
        self.layer4 = self._layer(512, 2, 2)
        self.linear = nn.Linear(512, 2)

    def _layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for step in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, step))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.bn1(self.conv1(x)))
        value = self.layer4(self.layer3(self.layer2(self.layer1(value))))
        return self.linear(F.adaptive_avg_pool2d(value, 1).flatten(1))


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x.mul(0.5).add(0.5).clamp(0, 1)


def _normalized(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 1).sub(0.5).div(0.5)


def adjust_gamma(x: torch.Tensor, gamma: float) -> torch.Tensor:
    return _normalized(_unit(x).pow(gamma))


def adjust_brightness(x: torch.Tensor, factor: float) -> torch.Tensor:
    return _normalized(_unit(x).mul(factor))


def adjust_contrast(x: torch.Tensor, factor: float) -> torch.Tensor:
    value = _unit(x)
    mean = value.mean(dim=(-2, -1), keepdim=True)
    return _normalized((value - mean).mul(factor).add(mean))


def affine(x: torch.Tensor, degrees: float = 0.0, translate_fraction: float = 0.0) -> torch.Tensor:
    angle = math.radians(degrees)
    batch = len(x)
    theta = x.new_zeros((batch, 2, 3))
    theta[:, 0, 0], theta[:, 0, 1] = math.cos(angle), -math.sin(angle)
    theta[:, 1, 0], theta[:, 1, 1] = math.sin(angle), math.cos(angle)
    # Historical parameter is in normalized grid units, not image-width fractions.
    # With align_corners=False, 0.03 shifts each axis by 1.5% of its image dimension.
    theta[:, 0, 2] = translate_fraction
    theta[:, 1, 2] = -translate_fraction
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def blur(x: torch.Tensor, strength: int = 3) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=strength, stride=1, padding=strength // 2)


def noise(x: torch.Tensor, sigma: float, generator: torch.Generator | None = None) -> torch.Tensor:
    usable_generator = generator if generator is not None and str(generator.device).startswith(x.device.type) else None
    draw = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=usable_generator)
    return (x + draw.mul(sigma / 0.5)).clamp(-1, 1)


def acquisition_view(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    value = x
    operations = int(torch.randint(1, 4, (1,), generator=generator).item())
    order = torch.randperm(7, generator=generator)[:operations].tolist()
    scalar = lambda low, high: low + (high - low) * float(torch.rand(1, generator=generator))
    for operation in order:
        if operation == 0:
            value = adjust_gamma(value, scalar(0.85, 1.15))
        elif operation == 1:
            value = adjust_brightness(value, scalar(0.90, 1.10))
        elif operation == 2:
            value = adjust_contrast(value, scalar(0.85, 1.15))
        elif operation == 3:
            value = affine(value, degrees=scalar(-7.0, 7.0))
        elif operation == 4:
            value = affine(value, translate_fraction=scalar(-0.03, 0.03))
        elif operation == 5:
            value = blur(value, 3)
        else:
            value = noise(value, scalar(0.005, 0.025), generator)
    return value


def jsd_loss(logits: list[torch.Tensor]) -> torch.Tensor:
    probabilities = [F.softmax(item, dim=1) for item in logits]
    mixture = torch.stack(probabilities).mean(0).clamp(1e-7, 1.0)
    return torch.stack([F.kl_div(mixture.log(), item, reduction="batchmean") for item in probabilities]).mean()


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    confidence = np.maximum(scores, 1 - scores)
    prediction = (scores >= 0.5).astype(np.int64)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for index in range(bins):
        selected = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if selected.any():
            total += selected.mean() * abs((prediction[selected] == labels[selected]).mean() - confidence[selected].mean())
    return float(total)


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    prediction = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {"n": int(len(labels)), "threshold": float(threshold),
            "accuracy": float(accuracy_score(labels, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
            "auroc": float(roc_auc_score(labels, scores)),
            "brier": float(brier_score_loss(labels, scores)),
            "ece10": expected_calibration_error(labels, scores),
            "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            transform=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if transform is not None:
                x = transform(x)
            logits.append(model(x).cpu())
            labels.append(y)
    logits_np = torch.cat(logits).numpy()
    labels_np = torch.cat(labels).numpy()
    scores = torch.softmax(torch.from_numpy(logits_np), 1).numpy()[:, 1]
    return logits_np, scores, labels_np


def gradcam(model: nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    saved = {}

    def hook(_module, _inputs, output):
        output.retain_grad()
        saved["activation"] = output

    handle = model.layer4.register_forward_hook(hook)
    try:
        value = x.to(device).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(value)
        logits[:, 1].sum().backward()
        activation = saved["activation"]
        weights = activation.grad.mean((2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.flatten(1).amax(1).clamp_min(1e-12).view(-1, 1, 1)
        return cam.detach().cpu().numpy()
    finally:
        handle.remove()


def explanation_proxies(model: nn.Module, images: np.ndarray, labels: np.ndarray,
                        device: torch.device, maximum: int = 64) -> dict:
    rng = np.random.default_rng(SEED + 17)
    chosen = np.concatenate([rng.choice(np.flatnonzero(labels == label), min(maximum // 2, int((labels == label).sum())), replace=False)
                             for label in (0, 1)])
    x = normalized_tensor(images[chosen])
    clean = gradcam(model, x, device)
    transforms = {
        "brightness_0p90": lambda value: adjust_brightness(value, 0.90),
        "contrast_0p85": lambda value: adjust_contrast(value, 0.85),
        "gamma_1p15": lambda value: adjust_gamma(value, 1.15),
        "blur_3": lambda value: blur(value, 3),
    }
    stability = {}
    flat_clean = clean.reshape(len(clean), -1)
    for name, transform in transforms.items():
        changed = gradcam(model, transform(x), device).reshape(len(clean), -1)
        cosine = (flat_clean * changed).sum(1) / (np.linalg.norm(flat_clean, axis=1) * np.linalg.norm(changed, axis=1) + 1e-12)
        stability[name] = {"mean_cam_cosine": float(cosine.mean()), "median_cam_cosine": float(np.median(cosine))}

    model.eval()
    with torch.no_grad():
        base = torch.softmax(model(x.to(device)), 1)[:, 1]
        height, width_image = x.shape[-2:]
        width = 6
        border_mask = torch.zeros((height, width_image), dtype=torch.bool)
        border_mask[:width, :] = True; border_mask[-width:, :] = True
        border_mask[:, :width] = True; border_mask[:, -width:] = True
        border_pixels = int(border_mask.sum())
        side = int(round(math.sqrt(border_pixels)))
        row0, col0 = (height - side) // 2, (width_image - side) // 2
        center_mask = torch.zeros_like(border_mask)
        center_mask[row0:row0 + side, col0:col0 + side] = True
        border = x.clone(); border[:, :, border_mask] = 0
        center = x.clone(); center[:, :, center_mask] = 0
        border_drop = base - torch.softmax(model(border.to(device)), 1)[:, 1]
        center_drop = base - torch.softmax(model(center.to(device)), 1)[:, 1]
    return {"n": int(len(chosen)), "target_class": 1, "cam_stability": stability,
            "mean_border_occlusion_drop": float(border_drop.mean().cpu()),
            "mean_center_occlusion_drop": float(center_drop.mean().cpu()),
            "center_minus_border_drop": float((center_drop - border_drop).mean().cpu()),
            "border_occluded_pixels": border_pixels,
            "center_occluded_pixels": int(center_mask.sum()),
            "interpretation_limit": "Source data have no lesion masks; these are stability and region-dependence proxies, not lesion-localization accuracy or medical causality."}


CORRUPTIONS = {
    "clean": None,
    "brightness_0p90": lambda x: adjust_brightness(x, 0.90),
    "brightness_1p10": lambda x: adjust_brightness(x, 1.10),
    "contrast_0p85": lambda x: adjust_contrast(x, 0.85),
    "contrast_1p15": lambda x: adjust_contrast(x, 1.15),
    "gamma_0p85": lambda x: adjust_gamma(x, 0.85),
    "gamma_1p15": lambda x: adjust_gamma(x, 1.15),
    "rotation_minus7": lambda x: affine(x, degrees=-7.0),
    "rotation_plus7": lambda x: affine(x, degrees=7.0),
    # Legacy result key retained for archived tables; actual per-axis translation is 1.5%.
    "translation_3pct": lambda x: affine(x, translate_fraction=0.03),
    "blur_3": lambda x: blur(x, 3),
    "noise_0p02": lambda x: noise(x, 0.02, torch.Generator(device=x.device).manual_seed(SEED + 99)),
}


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = SEED
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    milestones: tuple[int, int] = (50, 75)
    gamma: float = 0.1
    mode: str = "b64_plain"
    jsd_weight: float = 12.0


def prepare(root: Path, mode: str) -> Path:
    if mode not in {"b64_plain", "b64_augmix"}:
        raise ValueError(mode)
    dataset = obtain_data(root)
    arrays = load_arrays(dataset, include_test=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / "runs" / run_id / mode
    out.mkdir(parents=True)
    protocol = {"status": "protocol_frozen_before_training", "created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": "PneumoniaMNIST v2 64x64", "dataset_md5": md5(dataset), "dataset_sha256": sha256(dataset),
                "splits_loaded": {split: EXPECTED_SPLITS[split] for split in ("train", "val")},
                "test_loaded": False, "config": asdict(TrainingConfig(mode=mode)),
                "selection": "best clean validation AUROC within each mode; model comparison uses the frozen validation dual gate",
                "xai": "Grad-CAM stability plus matched border/center occlusion on source validation",
                "external_policy": "E_final was previously opened; it is unavailable for selection. A new untouched external holdout is required for unbiased final confirmation."}
    save_json(out / "protocol_before_training.json", protocol)
    return out


def train(root: Path, out: Path, mode: str) -> dict:
    config = TrainingConfig(mode=mode)
    seed_all(config.seed)
    arrays = load_arrays(obtain_data(root), include_test=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(config.milestones), gamma=config.gamma)
    train_loader = make_loader(arrays["train_images"], arrays["train_labels"], config.batch_size, True, config.seed)
    val_loader = make_loader(arrays["val_images"], arrays["val_labels"], config.batch_size, False, config.seed)
    augmentation_generator = torch.Generator().manual_seed(config.seed + 1)
    best_auc, best_epoch, rows = -np.inf, None, []
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train(); total_loss = total_ce = total_jsd = 0.0; count = correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if mode == "b64_augmix":
                view1 = acquisition_view(x, augmentation_generator)
                view2 = acquisition_view(x, augmentation_generator)
                outputs = [model(x), model(view1), model(view2)]
                ce = sum(F.cross_entropy(output, y) for output in outputs) / len(outputs)
                consistency = jsd_loss(outputs)
                loss = ce + config.jsd_weight * consistency
                logits = outputs[0]
            else:
                logits = model(x); ce = F.cross_entropy(logits, y); consistency = logits.new_tensor(0.0); loss = ce
            loss.backward(); optimizer.step()
            total_loss += float(loss) * len(y); total_ce += float(ce) * len(y); total_jsd += float(consistency) * len(y)
            count += len(y); correct += int((logits.argmax(1) == y).sum())
        val_logits, val_scores, val_labels = predict(model, val_loader, device)
        val = metrics(val_labels, val_scores)
        row = {"epoch": epoch, "learning_rate": float(optimizer.param_groups[0]["lr"]),
               "train_loss": total_loss / count, "train_ce": total_ce / count, "train_jsd": total_jsd / count,
               "train_accuracy": correct / count, "val_accuracy": val["accuracy"], "val_auroc": val["auroc"],
               "elapsed_seconds": time.time() - started}
        rows.append(row)
        if val["auroc"] > best_auc:
            best_auc, best_epoch = val["auroc"], epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "validation_auroc": best_auc,
                        "config": asdict(config)}, out / "best_weights.pt")
            np.savez_compressed(out / "best_validation_predictions.npz", logits=val_logits, scores=val_scores, labels=val_labels)
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(f"{mode} epoch={epoch:03d} val_auc={val['auroc']:.6f} val_acc={val['accuracy']:.6f}", flush=True)
    save_csv(out / "learning_curve.csv", rows)
    summary = {"status": "training_complete_validation_checkpoint_frozen", "mode": mode, "device": str(device),
               "best_epoch": int(best_epoch), "best_validation_auroc": float(best_auc),
               "weights_sha256": sha256(out / "best_weights.pt"), "training_seconds": time.time() - started}
    save_json(out / "training_summary.json", summary)
    return summary


def load_model(out: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(out / "best_weights.pt", map_location=device, weights_only=False)
    model = ResNet18().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def validate(root: Path, out: Path) -> dict:
    arrays = load_arrays(obtain_data(root), include_test=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(out, device)
    loader = make_loader(arrays["val_images"], arrays["val_labels"], 128, False, SEED)
    conditions = {}
    for name, transform in CORRUPTIONS.items():
        _, scores, labels = predict(model, loader, device, transform)
        conditions[name] = metrics(labels, scores)
    xai = explanation_proxies(model, arrays["val_images"], arrays["val_labels"], device)
    clean = conditions["clean"]
    perturbed = [value for key, value in conditions.items() if key != "clean"]
    summary = {"status": "validation_complete_test_unopened", "mode": json.loads((out / "protocol_before_training.json").read_text())["config"]["mode"],
               "weights_sha256": sha256(out / "best_weights.pt"), "conditions": conditions,
               "aggregate": {"clean_accuracy": clean["accuracy"], "clean_auroc": clean["auroc"],
                             "mean_perturbed_accuracy": float(np.mean([x["accuracy"] for x in perturbed])),
                             "mean_perturbed_auroc": float(np.mean([x["auroc"] for x in perturbed])),
                             "worst_perturbed_accuracy": float(np.min([x["accuracy"] for x in perturbed])),
                             "worst_perturbed_auroc": float(np.min([x["auroc"] for x in perturbed])),
                             "mean_perturbed_brier": float(np.mean([x["brier"] for x in perturbed])),
                             "mean_perturbed_ece10": float(np.mean([x["ece10"] for x in perturbed]))},
               "explainability": xai}
    save_json(out / "validation_robustness_and_xai.json", summary)
    return summary


def compare_validation(plain: dict, augmix: dict) -> dict:
    pa, aa = plain["aggregate"], augmix["aggregate"]
    px, ax = plain["explainability"], augmix["explainability"]
    p_cam = float(np.mean([x["mean_cam_cosine"] for x in px["cam_stability"].values()]))
    a_cam = float(np.mean([x["mean_cam_cosine"] for x in ax["cam_stability"].values()]))
    guardrails = {"clean_accuracy_not_lower_than_plain": aa["clean_accuracy"] >= pa["clean_accuracy"],
                  "clean_auroc_drop_le_0p005": aa["clean_auroc"] >= pa["clean_auroc"] - 0.005}
    benefits = {"mean_perturbed_auroc_higher": aa["mean_perturbed_auroc"] > pa["mean_perturbed_auroc"],
                "worst_perturbed_accuracy_higher": aa["worst_perturbed_accuracy"] > pa["worst_perturbed_accuracy"],
                "mean_cam_stability_higher": a_cam > p_cam,
                "center_minus_border_not_lower": ax["center_minus_border_drop"] >= px["center_minus_border_drop"]}
    accepted = all(guardrails.values()) and sum(benefits.values()) >= 2
    result = {"status": "candidate_selected_on_source_validation_only" if accepted else "candidate_rejected_on_source_validation",
              "guardrails": guardrails, "benefits": benefits, "accepted_for_internal_test": accepted,
              "differences_augmix_minus_plain": {key: float(aa[key] - pa[key]) for key in pa},
              "mean_cam_stability": {"b64_plain": float(p_cam), "b64_augmix": float(a_cam)},
              "test_access": "permitted once" if accepted else "forbidden"}
    return result


def evaluate_internal_test(root: Path, out: Path, selection: dict) -> dict:
    if not selection.get("accepted_for_internal_test", False):
        raise RuntimeError("Validation dual gate did not pass; internal test remains unopened for this candidate.")
    arrays = load_arrays(obtain_data(root), include_test=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(out, device)
    loader = make_loader(arrays["test_images"], arrays["test_labels"], 128, False, SEED)
    _, scores, labels = predict(model, loader, device)
    result = {"status": "one_time_internal_test_complete", "metrics": metrics(labels, scores),
              "explainability": explanation_proxies(model, arrays["test_images"], arrays["test_labels"], device),
              "comparators": {"B28": B28_TEST_REFERENCE, "CutMix_p0p5": CUTMIX_TEST_REFERENCE},
              "external_policy": "Do not reuse the previously opened E_final for selection; evaluate once on a newly sealed patient-disjoint external holdout after final freeze."}
    np.savez_compressed(out / "internal_test_predictions.npz", scores=scores, labels=labels)
    save_json(out / "internal_test_and_xai.json", result)
    return result
