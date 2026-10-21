"""Audit source dataset shape, target, and duplicate feature groups."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    data_path = ROOT / config["train_path"]
    frame = pd.read_csv(data_path)

    label = config["label_column"]
    excluded = set(config["excluded_feature_columns"])
    feature_columns = [column for column in frame.columns if column not in excluded]
    feature_frame = frame[feature_columns]

    # Hash only the model inputs. The full digest is retained to detect collisions
    # by comparing the source rows within each hashed group.
    row_hashes = pd.util.hash_pandas_object(feature_frame, index=False).astype("uint64")
    groups = row_hashes.groupby(row_hashes, sort=False).indices

    conflicting_label_groups = 0
    duplicate_groups = 0
    duplicate_extra_rows = 0
    largest_group = 0
    for positions in groups.values():
        size = len(positions)
        if size > 1:
            duplicate_groups += 1
            duplicate_extra_rows += size - 1
            largest_group = max(largest_group, size)
            if frame.iloc[positions][label].nunique(dropna=False) > 1:
                conflicting_label_groups += 1

    report = {
        "source_file": config["train_path"],
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "target": label,
        "target_counts": {
            str(key): int(value)
            for key, value in frame[label].value_counts(dropna=False).sort_index().items()
        },
        "excluded_feature_columns": sorted(excluded),
        "feature_columns": feature_columns,
        "feature_column_count": len(feature_columns),
        "unique_input_groups": len(groups),
        "duplicate_input_groups": duplicate_groups,
        "rows_beyond_first_per_duplicate_group": duplicate_extra_rows,
        "largest_duplicate_group": largest_group,
        "duplicate_groups_with_conflicting_labels": conflicting_label_groups,
        "input_group_hash": "pandas hash_pandas_object uint64 over feature columns; used for split grouping",
    }
    output_path = ROOT / "evidencias/auditoria.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nAuditoría guardada en {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
