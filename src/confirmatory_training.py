from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEED = 632901


@dataclass(frozen=True)
class Config:
    mode: str
    jsd_weight: float
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    milestones: tuple[int, int] = (50, 75)
    gamma: float = 0.1
    seed: int = SEED


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_split(dataset: Path, indices: Path) -> dict[str, np.ndarray]:
    data = np.load(dataset)
    split = np.load(indices)
    moved = split["moved_train_indices"]
    return {
        "train_images": data["train_images"][split["train_indices"]],
        "train_labels": data["train_labels"][split["train_indices"]].reshape(-1),
        "val_images": np.concatenate([data["val_images"], data["train_images"][moved]], axis=0),
        "val_labels": np.concatenate([data["val_labels"].reshape(-1), data["train_labels"][moved].reshape(-1)], axis=0),
    }


class ViewwiseBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm for a fused three-view forward with sequential-view semantics.

    Convolutions can process the concatenated clean/view1/view2 tensor in one
    launch.  During training, each BatchNorm layer still normalizes the three
    chunks independently and updates running statistics in clean, view1,
    view2 order, matching three consecutive model forwards.
    """

    split_three_views: bool = False

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training and self.split_three_views:
            if value.shape[0] % 3:
                raise ValueError(f"Three-view BatchNorm received batch size {value.shape[0]}")
            chunks = value.chunk(3, dim=0)
            return torch.cat([nn.BatchNorm2d.forward(self, chunk) for chunk in chunks], dim=0)
        return nn.BatchNorm2d.forward(self, value)


def install_viewwise_batchnorm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            replacement = ViewwiseBatchNorm2d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            ).to(device=child.weight.device if child.affine else child.running_mean.device)
            replacement.load_state_dict(child.state_dict())
            setattr(module, name, replacement)
        else:
            install_viewwise_batchnorm(child)


def set_three_view_batchnorm(module: nn.Module, enabled: bool) -> None:
    for child in module.modules():
        if isinstance(child, ViewwiseBatchNorm2d):
            child.split_three_views = enabled


def verify_viewwise_batchnorm(step15, device: torch.device) -> dict:
    """Numerically compare fused execution with three consecutive forwards."""
    torch.manual_seed(SEED + 404)
    reference = step15.ResNet18().to(device).train()
    fused = step15.ResNet18().to(device).train()
    fused.load_state_dict(reference.state_dict())
    install_viewwise_batchnorm(fused)
    views = [torch.randn(4, 1, 64, 64, device=device) for _ in range(3)]
    with torch.no_grad():
        expected = [reference(view) for view in views]
        set_three_view_batchnorm(fused, True)
        try:
            actual = list(fused(torch.cat(views, dim=0)).chunk(3, dim=0))
        finally:
            set_three_view_batchnorm(fused, False)
    output_max = max(float((left - right).abs().max()) for left, right in zip(expected, actual))
    state_max = 0.0
    for key, left in reference.state_dict().items():
        right = fused.state_dict()[key]
        if left.is_floating_point():
            state_max = max(state_max, float((left - right).abs().max()))
        elif not torch.equal(left, right):
            raise AssertionError(f"Non-floating BatchNorm state differs for {key}")
    # Fused convolutions can choose a different CUDA kernel than a four-image
    # forward, so allow only small floating-point roundoff.
    if output_max > 2e-4 or state_max > 2e-6:
        raise AssertionError({"output_max_abs_difference": output_max, "state_max_abs_difference": state_max})
    del reference, fused, views, expected, actual
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "status": "passed",
        "output_max_abs_difference": output_max,
        "state_max_abs_difference": state_max,
        "tolerance_output": 2e-4,
        "tolerance_state": 2e-6,
    }


def training_logits(model, x, y, mode: str, generator, step15, jsd_weight: float):
    if mode == "plain":
        logits = model(x)
        return logits, F.cross_entropy(logits, y), logits.new_tensor(0.0)
    if mode == "aug1":
        logits = model(step15.acquisition_view(x, generator))
        return logits, F.cross_entropy(logits, y), logits.new_tensor(0.0)
    view1 = step15.acquisition_view(x, generator)
    view2 = step15.acquisition_view(x, generator)
    # Convolutions use one concatenated forward, while ViewwiseBatchNorm2d
    # keeps the clean/view1/view2 statistics and running-stat update order of
    # the original three consecutive forwards.
    set_three_view_batchnorm(model, True)
    try:
        combined = model(torch.cat([x, view1, view2], dim=0))
    finally:
        set_three_view_batchnorm(model, False)
    outputs = list(combined.chunk(3, dim=0))
    ce = sum(F.cross_entropy(output, y) for output in outputs) / 3.0
    jsd = step15.jsd_loss(outputs)
    if mode == "ce3":
        return outputs[0], ce, jsd.detach()
    if mode == "acjsd":
        return outputs[0], ce + jsd_weight * jsd, jsd
    raise ValueError(mode)


def save_checkpoint(path: Path, model, epoch: int, config: Config, validation: dict) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epoch,
        "validation": validation,
        "config": asdict(config),
    }, path)


def run(dataset: Path, indices: Path, project_root: Path, output: Path, config: Config) -> dict:
    # The local archive keeps the module below the Chinese-named experiment
    # directory, whereas the canonical Colab notebook writes it directly below
    # /content/md6329_project.  Resolve both layouts explicitly so that the same
    # source file is executable in either environment.
    source_candidates = [
        project_root / "src",
        project_root / "md6329_step15_b64_augmix",
        project_root / "03_实验代码与数据" / "md6329_step15_b64_augmix",
    ]
    source = next((path for path in source_candidates if (path / "b64_augmix.py").is_file()), None)
    if source is None:
        raise FileNotFoundError(f"b64_augmix.py was not found under {source_candidates}")
    sys.path.insert(0, str(source))
    import b64_augmix as step15  # type: ignore

    output.mkdir(parents=True, exist_ok=False)
    arrays = load_split(dataset, indices)
    step15.seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keep the prespecified seed while using standard optimized CUDA kernels.
    # Exact backend flags are recorded because this is reproducible at the
    # protocol level but is not promised to be bitwise-identical across GPUs.
    if device.type == "cuda":
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    cuda_backend = {
        "bitwise_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    protocol = {
        "status": "source_only_protocol_frozen_before_training",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "dataset_sha256": sha256(dataset),
        "split_indices_sha256": sha256(indices),
        "checkpoint_policy": ["best_validation_brier", "best_validation_auroc", "epoch_40", "last_epoch"],
        "external_data_loaded": False,
        "resolved_source_module": str(source / "b64_augmix.py"),
        "cuda_backend": cuda_backend,
        "three_view_execution": "concatenated convolution with viewwise BatchNorm; clean/view1/view2 normalization and running-stat update order match three consecutive forwards",
        "precision": "CUDA AMP float16 forward/backward with FP32 master weights and GradScaler; FP32 on CPU",
    }
    (output / "protocol_before_training.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    model = step15.ResNet18().to(device)
    if config.mode in {"ce3", "acjsd"}:
        install_viewwise_batchnorm(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(config.milestones), gamma=config.gamma)
    shuffle_generator = torch.Generator().manual_seed(config.seed)
    augmentation_generator = torch.Generator().manual_seed(config.seed + 1)
    train_loader = DataLoader(
        TensorDataset(step15.normalized_tensor(arrays["train_images"]), torch.from_numpy(arrays["train_labels"]).long()),
        batch_size=config.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = step15.make_loader(arrays["val_images"], arrays["val_labels"], config.batch_size, False, config.seed)
    best_auc = -np.inf
    best_brier = np.inf
    best_auc_epoch = None
    best_brier_epoch = None
    rows: list[dict] = []
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        count = correct = 0
        total_loss = total_jsd = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, loss, jsd = training_logits(model, x, y, config.mode, augmentation_generator, step15, config.jsd_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count += len(y)
            correct += int((logits.argmax(1) == y).sum())
            total_loss += float(loss.detach()) * len(y)
            total_jsd += float(jsd.detach()) * len(y)
        _, val_scores, val_labels = step15.predict(model, val_loader, device)
        val = step15.metrics(val_labels, val_scores)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total_loss / count,
            "train_jsd": total_jsd / count,
            "train_accuracy": correct / count,
            "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_auroc": val["auroc"],
            "val_brier": val["brier"],
            "elapsed_seconds": time.time() - started,
        }
        rows.append(row)
        if val["auroc"] > best_auc:
            best_auc = float(val["auroc"])
            best_auc_epoch = epoch
            save_checkpoint(output / "best_validation_auroc.pt", model, epoch, config, val)
        if val["brier"] < best_brier:
            best_brier = float(val["brier"])
            best_brier_epoch = epoch
            save_checkpoint(output / "best_validation_brier.pt", model, epoch, config, val)
        if epoch == 40:
            save_checkpoint(output / "epoch_40.pt", model, epoch, config, val)
        scheduler.step()
        save_csv(output / "learning_curve.csv", rows)
        if epoch == 1 or epoch % 10 == 0:
            print(f"{config.mode} lambda={config.jsd_weight:g} epoch={epoch:03d} val_auc={val['auroc']:.6f} val_brier={val['brier']:.6f}", flush=True)
    save_checkpoint(output / "last_epoch.pt", model, config.epochs, config, val)
    summary = {
        "status": "source_training_complete_external_unopened",
        "config": asdict(config),
        "best_validation_auroc": best_auc,
        "best_validation_auroc_epoch": best_auc_epoch,
        "best_validation_brier": best_brier,
        "best_validation_brier_epoch": best_brier_epoch,
        "checkpoint_sha256": {path.stem: sha256(path) for path in output.glob("*.pt")},
        "training_seconds": time.time() - started,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_backend": cuda_backend,
        "three_view_execution": protocol["three_view_execution"],
        "precision": protocol["precision"],
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["plain", "aug1", "ce3", "acjsd"], required=True)
    parser.add_argument("--jsd-weight", type=float, default=12.0)
    args = parser.parse_args()
    if args.mode != "acjsd" and args.jsd_weight != 12.0:
        raise ValueError("JSD weight only applies to acjsd")
    tag = f"{args.mode}_lambda_{args.jsd_weight:g}" if args.mode == "acjsd" else args.mode
    run(args.dataset, args.indices, args.project_root, args.output_root / tag, Config(args.mode, args.jsd_weight))


if __name__ == "__main__":
    main()
