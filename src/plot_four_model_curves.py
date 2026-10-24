from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def embedded_step18(notebook: Path) -> dict:
    value = None
    data = json.loads(notebook.read_text(encoding="utf-8"))
    for cell in data["cells"]:
        source = "".join(cell.get("source", []))
        if "STEP18_RECORDED" not in source or cell.get("cell_type") != "code":
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "STEP18_RECORDED" for target in node.targets):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    if isinstance(node.value, ast.Call) and node.value.args:
                        value = json.loads(ast.literal_eval(node.value.args[0]))
                    else:
                        raise
    if value is None:
        raise RuntimeError("STEP18_RECORDED was not found")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--plain-curve", type=Path, required=True)
    parser.add_argument("--acjsd-curve", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    recorded = embedded_step18(args.notebook)
    curves = {
        "Plain": read_csv(args.plain_curve),
        "Aug1": recorded["models"]["b64_aug1"]["curve"],
        "CE3": recorded["models"]["b64_ce3_jsd0_full"]["curve"],
        "AC-JSD": read_csv(args.acjsd_curve),
    }
    selected = {"Plain": 18, "Aug1": 61, "CE3": 36, "AC-JSD": 40}
    colors = {"Plain": "#666666", "Aug1": "#b17b28", "CE3": "#925a80", "AC-JSD": "#24738b"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained")
    for name, rows in curves.items():
        epochs = [int(row["epoch"]) for row in rows]
        axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label=name, color=colors[name], linewidth=1.2)
        axes[1].plot(epochs, [float(row["val_auroc"]) for row in rows], label=name, color=colors[name], linewidth=1.2)
        point = next(row for row in rows if int(row["epoch"]) == selected[name])
        axes[1].scatter(selected[name], float(point["val_auroc"]), color=colors[name], s=24, zorder=3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Training objective (log scale)")
    axes[1].set_ylabel("Clean validation AUROC")
    axes[1].set_ylim(0.975, 1.001)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=7, ncol=2)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"))
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=180)
    plt.close(fig)
    manifest = {
        "curves": {name: len(rows) for name, rows in curves.items()},
        "selected_epochs": selected,
        "note": "AC-JSD's archived curve ends at its selected epoch 40; no later AC-JSD rows were available locally.",
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
