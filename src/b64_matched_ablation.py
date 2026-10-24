from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SEED = 632901
EPOCHS = 100
MODES = {"b64_aug1", "b64_ce3_jsd0_full"}


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def config_for(mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(mode)
    return {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": 128,
        "learning_rate": 0.001,
        "milestones": [50, 75],
        "gamma": 0.1,
        "mode": mode,
        "jsd_weight": 0.0,
        "checkpoint_selection": "best clean validation AUROC",
    }


def prepare(stage_root: Path, data_root: Path, mode: str, step15) -> Path:
    stage_root, data_root = Path(stage_root), Path(data_root)
    config = config_for(mode)
    dataset = step15.obtain_data(data_root)
    step15.load_arrays(dataset, include_test=False)
    out = stage_root / mode
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "matched_ablation_protocol_frozen_before_training",
        "dataset": "PneumoniaMNIST v2 64x64",
        "dataset_md5": step15.md5(dataset),
        "dataset_sha256": step15.sha256(dataset),
        "splits_loaded": {"train": 4708, "val": 524},
        "test_loaded_before_training": False,
        "config": config,
        "shared_controls": {
            "architecture": "same B64 ResNet18",
            "optimizer": "Adam",
            "schedule": "milestones 50/75, gamma 0.1",
            "augmentation_function": "the frozen Step 15 acquisition_view",
            "seed": SEED,
            "epoch_budget": EPOCHS,
            "checkpoint_rule": "best clean validation AUROC",
        },
        "estimand": (
            "b64_aug1 minus B64 plain estimates ordinary one-view augmentation; "
            "b64_ce3_jsd0_full minus b64_aug1 estimates the incremental three-view supervised objective; "
            "full AC-JSD minus b64_ce3_jsd0_full estimates the JSD increment"
        ),
        "internal_test_policy": (
            "evaluate after each 100-epoch checkpoint is frozen; the test was already known, "
            "so this remains retrospective attribution rather than confirmatory selection"
        ),
        "external_policy": "no external data are used in this matched source ablation",
    }
    protocol_path = out / "protocol_before_training.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("config") != config:
            raise RuntimeError(f"Existing protocol differs for {mode}: {protocol_path}")
    else:
        save_json(protocol_path, protocol)
    return out


def _restore_rng(state: dict, shuffle_generator: torch.Generator, augmentation_generator: torch.Generator) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    shuffle_generator.set_state(state["shuffle_generator_state"])
    augmentation_generator.set_state(state["augmentation_generator_state"])


def _resume_payload(model, optimizer, scheduler, next_epoch, best_auc, best_epoch, rows,
                    config, shuffle_generator, augmentation_generator) -> dict:
    return {
        "status": "epoch_boundary_resume_state",
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "next_epoch": next_epoch,
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "rows": rows,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "shuffle_generator_state": shuffle_generator.get_state(),
        "augmentation_generator_state": augmentation_generator.get_state(),
    }


def train(data_root: Path, out: Path, mode: str, step15, resume: bool = True) -> dict:
    data_root, out = Path(data_root), Path(out)
    config = config_for(mode)
    step15.seed_all(SEED)
    arrays = step15.load_arrays(step15.obtain_data(data_root), include_test=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = step15.ResNet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config["milestones"], gamma=config["gamma"]
    )
    shuffle_generator = torch.Generator().manual_seed(SEED)
    augmentation_generator = torch.Generator().manual_seed(SEED + 1)
    rows, best_auc, best_epoch, start_epoch = [], -np.inf, None, 1
    resume_path = out / "resume_state.pt"
    if resume and resume_path.exists():
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        if state.get("config") != config:
            raise RuntimeError(f"Resume config mismatch: {resume_path}")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["next_epoch"])
        best_auc = float(state["best_auc"])
        best_epoch = int(state["best_epoch"]) if state["best_epoch"] is not None else None
        rows = list(state["rows"])
        _restore_rng(state, shuffle_generator, augmentation_generator)
        print(f"Resuming {mode} from epoch {start_epoch}", flush=True)

    train_dataset = TensorDataset(
        step15.normalized_tensor(arrays["train_images"]),
        torch.from_numpy(arrays["train_labels"]).long(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=shuffle_generator,
    )
    val_loader = step15.make_loader(
        arrays["val_images"], arrays["val_labels"], config["batch_size"], False, SEED
    )
    started = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = total_jsd = 0.0
        count = correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if mode == "b64_aug1":
                augmented = step15.acquisition_view(x, augmentation_generator)
                logits = model(augmented)
                loss = F.cross_entropy(logits, y)
                diagnostic_jsd = logits.new_tensor(0.0)
            else:
                view1 = step15.acquisition_view(x, augmentation_generator)
                view2 = step15.acquisition_view(x, augmentation_generator)
                outputs = [model(x), model(view1), model(view2)]
                losses = [F.cross_entropy(output, y) for output in outputs]
                loss = sum(losses) / len(losses)
                diagnostic_jsd = step15.jsd_loss(outputs).detach()
                logits = outputs[0]
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            total_jsd += float(diagnostic_jsd) * len(y)
            count += len(y)
            correct += int((logits.argmax(1) == y).sum())

        val_logits, val_scores, val_labels = step15.predict(model, val_loader, device)
        val = step15.metrics(val_labels, val_scores)
        rows.append({
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total_loss / count,
            "diagnostic_jsd": total_jsd / count,
            "train_accuracy": correct / count,
            "val_accuracy": val["accuracy"],
            "val_auroc": val["auroc"],
            "elapsed_seconds_this_session": time.time() - started,
        })
        if val["auroc"] > best_auc:
            best_auc, best_epoch = float(val["auroc"]), epoch
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "validation_auroc": best_auc,
                "config": config,
            }, out / "best_weights.pt")
            np.savez_compressed(
                out / "best_validation_predictions.npz",
                logits=val_logits,
                scores=val_scores,
                labels=val_labels,
            )
        scheduler.step()
        torch.save(
            _resume_payload(
                model, optimizer, scheduler, epoch + 1, best_auc, best_epoch, rows,
                config, shuffle_generator, augmentation_generator,
            ),
            resume_path,
        )
        step15.save_csv(out / "learning_curve.csv", rows)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"{mode} epoch={epoch:03d}/{EPOCHS} "
                f"val_auc={val['auroc']:.6f} val_acc={val['accuracy']:.6f}",
                flush=True,
            )

    completed_epochs = len(rows)
    if completed_epochs != EPOCHS or not (out / "best_weights.pt").exists():
        raise RuntimeError(f"Matched run incomplete: {mode}, epochs={completed_epochs}")
    summary = {
        "status": "matched_100_epoch_training_complete_checkpoint_frozen",
        "mode": mode,
        "planned_epochs": EPOCHS,
        "completed_epochs": completed_epochs,
        "best_epoch": int(best_epoch),
        "best_validation_auroc": float(best_auc),
        "weights_sha256": step15.sha256(out / "best_weights.pt"),
        "resume_state_saved": True,
        "session_seconds": time.time() - started,
    }
    save_json(out / "training_summary.json", summary)
    return summary


def validate(data_root: Path, out: Path, step15) -> dict:
    summary = step15.validate(Path(data_root), Path(out))
    summary["matched_training_complete"] = (
        json.loads((Path(out) / "training_summary.json").read_text(encoding="utf-8"))
        .get("completed_epochs") == EPOCHS
    )
    if not summary["matched_training_complete"]:
        raise RuntimeError("Refusing to validate an incomplete matched run")
    save_json(Path(out) / "validation_robustness_and_xai.json", summary)
    return summary


def evaluate_internal(data_root: Path, out: Path, label: str, step15) -> dict:
    training = json.loads((Path(out) / "training_summary.json").read_text(encoding="utf-8"))
    if training.get("completed_epochs") != EPOCHS:
        raise RuntimeError("Internal evaluation requires an exact completed 100-epoch run")
    arrays = step15.load_arrays(step15.obtain_data(Path(data_root)), include_test=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = step15.load_model(Path(out), device)
    loader = step15.make_loader(arrays["test_images"], arrays["test_labels"], 128, False, SEED)
    logits, scores, labels = step15.predict(model, loader, device)
    result = {
        "status": "retrospective_matched_internal_test_complete",
        "model": label,
        "metrics": step15.metrics(labels, scores),
        "explainability": step15.explanation_proxies(
            model, arrays["test_images"], arrays["test_labels"], device
        ),
        "interpretation": (
            "The source test was already known before Step 18. Use this result for matched attribution, "
            "not confirmatory model selection or significance claims."
        ),
    }
    np.savez_compressed(
        Path(out) / "internal_test_predictions_step18.npz",
        logits=logits,
        scores=scores,
        labels=labels,
    )
    save_json(Path(out) / "internal_test_and_xai_step18.json", result)
    return result


def compact_row(name: str, validation: dict, internal: dict, weights_sha256: str) -> dict:
    va, vm = validation["aggregate"], internal["metrics"]
    vx, ix = validation["explainability"], internal["explainability"]
    return {
        "model": name,
        "weights_sha256": weights_sha256,
        "val_clean_accuracy": va["clean_accuracy"],
        "val_clean_auroc": va["clean_auroc"],
        "val_mean_perturbed_auroc": va["mean_perturbed_auroc"],
        "val_worst_perturbed_accuracy": va["worst_perturbed_accuracy"],
        "val_mean_cam_stability": float(np.mean([
            item["mean_cam_cosine"] for item in vx["cam_stability"].values()
        ])),
        "val_center_minus_border_drop": vx["center_minus_border_drop"],
        "test_accuracy": vm["accuracy"],
        "test_balanced_accuracy": vm["balanced_accuracy"],
        "test_auroc": vm["auroc"],
        "test_brier": vm["brier"],
        "test_sensitivity": vm["sensitivity"],
        "test_specificity": vm["specificity"],
        "test_mean_cam_stability": float(np.mean([
            item["mean_cam_cosine"] for item in ix["cam_stability"].values()
        ])),
        "test_center_minus_border_drop": ix["center_minus_border_drop"],
    }


def delta(new: dict, old: dict) -> dict:
    return {
        key: float(new[key] - old[key])
        for key in new
        if key not in {"model", "weights_sha256"} and isinstance(new[key], (int, float))
    }


def smoke_test() -> dict:
    return {
        "modes": sorted(MODES),
        "epochs": EPOCHS,
        "resume_state": ["model", "optimizer", "scheduler", "rng", "sampler", "augmentation"],
    }
