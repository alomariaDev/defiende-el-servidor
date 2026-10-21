"""Audit a CSV and document exact duplicate predictor groups."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_summary(frame: pd.DataFrame, predictors: list[str], label: str) -> dict:
    if not predictors:
        raise ValueError("No quedan variables predictoras después de excluir id, label y attack_cat")

    predictor_frame = frame[predictors]
    fingerprints = pd.util.hash_pandas_object(predictor_frame, index=False).astype("uint64")
    groups = fingerprints.groupby(fingerprints, sort=False).indices
    duplicate_groups = 0
    duplicate_extra_rows = 0
    largest_group = 1
    contradictory_groups = 0
    for positions in groups.values():
        size = len(positions)
        if size < 2:
            continue
        duplicate_groups += 1
        duplicate_extra_rows += size - 1
        largest_group = max(largest_group, size)
        if frame.iloc[positions][label].nunique(dropna=False) > 1:
            contradictory_groups += 1

    return {
        "predictor_columns": predictors,
        "predictor_column_count": len(predictors),
        "unique_predictor_groups": int(len(groups)),
        "duplicate_predictor_groups": int(duplicate_groups),
        "rows_beyond_first_in_duplicate_groups": int(duplicate_extra_rows),
        "largest_duplicate_group": int(largest_group),
        "duplicate_groups_with_contradictory_labels": int(contradictory_groups),
        "grouping_method": "uint64 fingerprints from pandas hash_pandas_object over predictor columns; same fingerprints are kept in one split group",
    }


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        default=config["train_path"],
        help="CSV to audit (default: configured training CSV)",
    )
    parser.add_argument(
        "--output",
        default="evidencias/auditoria.json",
        help="JSON report destination",
    )
    args = parser.parse_args()

    data_path = Path(args.csv).resolve()
    frame = pd.read_csv(data_path)
    label = config["label_column"]
    excluded_names = {"id", "label", "attack_cat"}
    predictors = [column for column in frame.columns if column.lower() not in excluded_names]

    report = {
        "source_file": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
        "sha256": sha256_file(data_path),
        "rows": int(len(frame)),
        "columns_count": int(len(frame.columns)),
        "columns": list(frame.columns),
        "types": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values_by_column": {
            column: int(count) for column, count in frame.isna().sum().items()
        },
        "missing_values_total": int(frame.isna().sum().sum()),
        "target": label if label in frame.columns else None,
        "class_counts": (
            {str(key): int(value) for key, value in frame[label].value_counts(dropna=False).sort_index().items()}
            if label in frame.columns
            else None
        ),
        "excluded_from_predictors": [column for column in frame.columns if column.lower() in excluded_names],
        "duplicate_input_audit": duplicate_summary(frame, predictors, label) if label in frame.columns else None,
        "contradictory_label_policy": (
            "Retain all rows and labels; keep rows with the same predictor fingerprint in one split. "
            "Report contradictory-label fingerprint groups without resolving labels or deleting rows."
        ),
    }
    output_path = (ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nAuditoría guardada en {output_path}")


if __name__ == "__main__":
    main()
