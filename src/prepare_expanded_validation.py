from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 632901


def digest(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset)
    train_x = data["train_images"]
    train_y = data["train_labels"].reshape(-1).astype(int)
    val_x = data["val_images"]
    val_y = data["val_labels"].reshape(-1).astype(int)
    test_x = data["test_images"]

    val_hashes = {digest(x) for x in val_x}
    test_hashes = {digest(x) for x in test_x}
    seen: set[str] = set()
    eligible: list[tuple[int, int, str]] = []
    excluded_within_train = 0
    excluded_train_val = 0
    excluded_train_test = 0
    for index, (image, label) in enumerate(zip(train_x, train_y)):
        image_hash = digest(image)
        if image_hash in seen:
            excluded_within_train += 1
            continue
        seen.add(image_hash)
        if image_hash in val_hashes:
            excluded_train_val += 1
            continue
        if image_hash in test_hashes:
            excluded_train_test += 1
            continue
        eligible.append((index, int(label), image_hash))

    moved: list[int] = []
    retained: list[int] = []
    per_class: dict[str, dict[str, int]] = {}
    for label in sorted(set(train_y.tolist())):
        group = [row for row in eligible if row[1] == label]
        group.sort(key=lambda row: hashlib.sha256(f"{SEED}|{label}|{row[2]}".encode()).hexdigest())
        n_move = math.ceil(0.10 * len(group))
        moved.extend(row[0] for row in group[:n_move])
        retained.extend(row[0] for row in group[n_move:])
        per_class[str(label)] = {"eligible": len(group), "moved": n_move, "retained": len(group) - n_move}

    moved = sorted(moved)
    retained = sorted(retained)
    expanded_val_labels = np.concatenate([val_y, train_y[moved]])
    np.savez_compressed(
        args.output_dir / "expanded_validation_indices.npz",
        train_indices=np.asarray(retained, dtype=np.int64),
        original_val_indices=np.arange(len(val_y), dtype=np.int64),
        moved_train_indices=np.asarray(moved, dtype=np.int64),
    )
    manifest = {
        "status": "prepared_source_only_no_training",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "seed": SEED,
        "method": "exact-image SHA-256 deduplication followed by class-stratified deterministic 10% SHA-rank transfer",
        "excluded": {
            "within_train_duplicate_images": excluded_within_train,
            "train_images_duplicated_in_original_validation": excluded_train_val,
            "train_images_duplicated_in_test": excluded_train_test,
        },
        "original": {"train": len(train_y), "validation": len(val_y), "test": len(test_x)},
        "new": {
            "train": len(retained),
            "validation": len(expanded_val_labels),
            "moved_from_train": len(moved),
            "validation_class_counts": {str(k): int(v) for k, v in Counter(expanded_val_labels.tolist()).items()},
        },
        "per_class_train_split": per_class,
        "limitations": [
            "PneumoniaMNIST does not expose patient identifiers, so exact-image groups rather than patients define the deduplication unit.",
            "The official source test is unchanged and is not used for selection."
        ],
    }
    (args.output_dir / "expanded_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
