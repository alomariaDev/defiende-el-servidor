"""Create a fixed-seed, duplicate-group-safe train/validation split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=config["train_path"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--validation-fraction", type=float, default=config["validation_fraction"])
    args = parser.parse_args()

    source_path = Path(args.csv).resolve()
    frame = pd.read_csv(source_path)
    label = config["label_column"]
    excluded = {"id", "label", "attack_cat"}
    predictors = [column for column in frame.columns if column.lower() not in excluded]
    groups = pd.util.hash_pandas_object(frame[predictors], index=False).astype("uint64")

    n_splits = round(1 / args.validation_fraction)
    if n_splits < 2 or abs((1 / n_splits) - args.validation_fraction) > 1e-9:
        raise ValueError("validation-fraction debe ser el recíproco de un entero >= 2")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    train_positions, validation_positions = next(
        splitter.split(frame[predictors], frame[label], groups=groups)
    )
    train_groups = set(groups.iloc[train_positions].tolist())
    validation_groups = set(groups.iloc[validation_positions].tolist())
    overlap = train_groups & validation_groups
    if overlap:
        raise RuntimeError(f"Fuga de grupos: {len(overlap)} grupos cruzan train/val")

    feature_columns = [column for column in frame.columns if column.lower() not in excluded]
    output_columns = (["id"] if "id" in frame.columns else []) + feature_columns + [label]
    train = frame.iloc[train_positions][output_columns].copy()
    validation = frame.iloc[validation_positions][output_columns].copy()
    if "id" in train.columns:
        train.rename(columns={"id": "row_id"}, inplace=True)
        validation.rename(columns={"id": "row_id"}, inplace=True)

    output_dir = ROOT / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    validation.to_csv(output_dir / "val.csv", index=False)

    indices = pd.concat(
        [
            pd.DataFrame({"source_row_index": train_positions, "split": "train"}),
            pd.DataFrame({"source_row_index": validation_positions, "split": "validation"}),
        ],
        ignore_index=True,
    ).sort_values("source_row_index", kind="stable")
    if "id" in frame.columns:
        indices.insert(1, "row_id", frame.iloc[indices["source_row_index"].to_numpy()]["id"].to_numpy())
    indices.to_csv(output_dir / "split_indices.csv", index=False)

    report = {
        "source_file": str(source_path.relative_to(ROOT)) if source_path.is_relative_to(ROOT) else str(source_path),
        "seed": args.seed,
        "method": "StratifiedGroupKFold; first fold; equal uint64 predictor fingerprints stay together",
        "requested_validation_fraction": args.validation_fraction,
        "train_rows": int(len(train_positions)),
        "validation_rows": int(len(validation_positions)),
        "actual_validation_fraction": float(len(validation_positions) / len(frame)),
        "train_target_counts": {str(k): int(v) for k, v in frame.iloc[train_positions][label].value_counts().sort_index().items()},
        "validation_target_counts": {str(k): int(v) for k, v in frame.iloc[validation_positions][label].value_counts().sort_index().items()},
        "train_unique_input_groups": int(len(train_groups)),
        "validation_unique_input_groups": int(len(validation_groups)),
        "overlapping_input_groups": int(len(overlap)),
        "indices_file": "data/processed/split_indices.csv",
        "indices_are_zero_based_source_row_positions": True,
    }
    (ROOT / "evidencias/split.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Archivos guardados: data/processed/train.csv, val.csv y split_indices.csv")


if __name__ == "__main__":
    main()
