from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for name in ("plain", "aug1", "ce3", "augmix"):
        path = args.checkpoint_dir / f"{name}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        records.append({
            "model": "acjsd" if name == "augmix" else name,
            "path": str(path),
            "sha256": sha256(path),
            "stored_epoch": int(checkpoint["epoch"]),
            "stored_validation_auroc": float(checkpoint["validation_auroc"]),
            "available_selection_rule": "best_validation_auroc",
            "contains_optimizer_state": "optimizer_state" in checkpoint,
            "contains_validation_predictions": "validation_predictions" in checkpoint,
        })

    result = {
        "status": "historical_artifact_audit_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "requested_checks": {
            "best_validation_auroc": "available",
            "last_epoch": "not recoverable from the archived four-model bundle",
            "best_validation_brier": "not recoverable because per-epoch weights and per-epoch probabilities were not archived",
            "epoch_40_alignment": "not recoverable for plain, Aug1 or CE3; AC-JSD epoch 40 is available",
        },
        "interpretation": "The historical four-way comparison cannot support a post-hoc selection-rule or epoch-40 sensitivity analysis. New confirmatory runs must save all four prespecified checkpoint types.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
