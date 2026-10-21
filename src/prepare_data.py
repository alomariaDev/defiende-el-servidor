"""Create a reproducible, group-safe train/validation split from training data only."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    frame = pd.read_csv(ROOT / config["train_path"])
    label = config["label_column"]
    excluded = set(config["excluded_feature_columns"])
    features = [column for column in frame.columns if column not in excluded]
    groups = pd.util.hash_pandas_object(frame[features], index=False).astype("uint64")
    y = frame[label]

    # Use the configured validation fraction as a fold count when it maps exactly
    # to a practical integer number of folds; for 0.2 this yields five folds.
    n_splits = round(1 / config["validation_fraction"])
    if n_splits < 2 or abs((1 / n_splits) - config["validation_fraction"]) > 1e-9:
        raise ValueError("validation_fraction must be the reciprocal of an integer >= 2")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config["seed"],
    )
    train_positions, validation_positions = next(
        splitter.split(frame[features], y, groups=groups)
    )
    train = frame.iloc[train_positions].copy()
    validation = frame.iloc[validation_positions].copy()

    train_groups = set(groups.iloc[train_positions].tolist())
    validation_groups = set(groups.iloc[validation_positions].tolist())
    overlap = train_groups & validation_groups
    if overlap:
        raise RuntimeError(f"Split leakage: {len(overlap)} duplicate-input groups cross splits")

    # Keep id only as row_id for provenance; it is never a model feature.
    output_columns = ["id", *features, label]
    train = train[output_columns].rename(columns={"id": "row_id"})
    validation = validation[output_columns].rename(columns={"id": "row_id"})
    output_dir = ROOT / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    validation.to_csv(output_dir / "val.csv", index=False)

    report = {
        "seed": config["seed"],
        "method": "StratifiedGroupKFold; first fold; groups are hashes of model-input columns",
        "requested_validation_fraction": config["validation_fraction"],
        "train_rows": len(train),
        "validation_rows": len(validation),
        "actual_validation_fraction": len(validation) / len(frame),
        "train_target_counts": {str(k): int(v) for k, v in train[label].value_counts().sort_index().items()},
        "validation_target_counts": {str(k): int(v) for k, v in validation[label].value_counts().sort_index().items()},
        "train_unique_input_groups": len(train_groups),
        "validation_unique_input_groups": len(validation_groups),
        "overlapping_input_groups": len(overlap),
        "files": ["data/processed/train.csv", "data/processed/val.csv"],
    }
    report_path = ROOT / "evidencias/split.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
