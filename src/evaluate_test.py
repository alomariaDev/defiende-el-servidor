"""Evaluate the frozen v2 model once on the complete UNSW-NB15 testing set."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TEST_ROWS = 82_332
TEST_PATH = ROOT / "data/raw/UNSW_NB15_testing-set.csv"
TRAIN_PATH = ROOT / "data/processed/train.csv"
VAL_PATH = ROOT / "data/processed/val.csv"
MODEL_PATH = ROOT / "evidencias/huggingface/v2/model/model.skops"
SELECTION_PATH = ROOT / "evidencias/seleccion.json"
RECOVERY_MANIFEST_PATH = ROOT / "evidencias/recuperacion/manifest.json"
PREDICTIONS_PATH = ROOT / "evidencias/predicciones-test.csv"
METRICS_PATH = ROOT / "evidencias/metricas-test.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predictor_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"id", "row_id", "label", "attack_cat"}
    return [column for column in frame.columns if column.lower() not in excluded]


def canonical_predictors(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalize numeric dtypes so int/float CSV inference does not hide equal values."""
    normalized = {}
    for column in columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            normalized[column] = pd.to_numeric(series, errors="raise").astype("float64")
        else:
            normalized[column] = series.astype("string")
    return pd.DataFrame(normalized, index=frame.index)


def fingerprints(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    canonical = canonical_predictors(frame, columns)
    return pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype="uint64")


def main() -> None:
    if PREDICTIONS_PATH.exists() or METRICS_PATH.exists():
        raise FileExistsError(
            "Ya existen evidencias de evaluación test; no se repetirá la inferencia. "
            "Inspecciona esos archivos antes de continuar."
        )

    selection = json.loads(SELECTION_PATH.read_text())
    if selection["version_selected"] != "v2":
        raise RuntimeError("seleccion.json ya no selecciona v2; evaluación cancelada")
    threshold = float(selection["single_change"]["to_v2"])
    expected_model_sha = selection["model"]["artifact_sha256"]
    if selection["single_change"].get("model_and_preprocessing_unchanged") is not True:
        raise RuntimeError("La selección ya no describe el modelo congelado esperado")

    hub_evidence = json.loads((ROOT / "evidencias/huggingface.json").read_text())
    hub_revision = hub_evidence["revisions"]["v2"]["commit_sha"]
    hub_model_git_commit = hub_evidence["revisions"]["v2"]["git_commit"]
    selection_hub_sha = selection.get("huggingface", {}).get("v2_commit_sha")
    if hub_revision != selection_hub_sha:
        raise RuntimeError("El SHA v2 de Hugging Face no coincide con seleccion.json")
    hub_config = json.loads((ROOT / "evidencias/huggingface/v2/config.json").read_text())
    if float(hub_config["decision_threshold"]) != threshold:
        raise RuntimeError("El umbral de seleccion.json difiere de la configuración congelada v2")
    if hub_evidence["revisions"]["v2"]["git_tag"] != selection["git_reference"]["freeze_tag"]:
        raise RuntimeError("La etiqueta Git de Hugging Face difiere de seleccion.json")
    model_sha = sha256(MODEL_PATH)
    if model_sha != expected_model_sha or model_sha != hub_evidence["revisions"]["v2"]["model_sha256"]:
        raise RuntimeError(f"El SHA del Pipeline congelado no coincide: {model_sha}")

    recovery_manifest = json.loads(RECOVERY_MANIFEST_PATH.read_text())
    expected_test_sha = recovery_manifest["data_sha256"]["data/raw/UNSW_NB15_testing-set.csv"]
    test_sha = sha256(TEST_PATH)
    if test_sha != expected_test_sha:
        raise RuntimeError(f"El SHA-256 del testing-set no coincide con recuperación: {test_sha}")

    # This is the sole parse of the test CSV and the sole inference call.
    test = pd.read_csv(TEST_PATH)
    if len(test) != EXPECTED_TEST_ROWS:
        raise RuntimeError(f"Se esperaban {EXPECTED_TEST_ROWS} filas y se leyeron {len(test)}")
    if "id" not in test or "label" not in test:
        raise RuntimeError("El testing-set debe incluir id y label")

    columns = predictor_columns(test)
    if not columns:
        raise RuntimeError("No se encontraron variables predictoras")
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VAL_PATH)
    if set(columns) != set(predictor_columns(train)) or set(columns) != set(predictor_columns(validation)):
        raise RuntimeError("Las columnas predictoras del test no coinciden con train/val")

    test_fingerprints = fingerprints(test, columns)
    train_fingerprints = fingerprints(train, columns)
    val_fingerprints = fingerprints(validation, columns)
    train_groups = set(train_fingerprints.tolist())
    val_groups = set(val_fingerprints.tolist())
    test_groups = set(test_fingerprints.tolist())
    train_overlap_groups = test_groups & train_groups
    val_overlap_groups = test_groups & val_groups
    both_overlap_groups = train_overlap_groups & val_overlap_groups
    test_matches_train = np.isin(test_fingerprints, list(train_groups))
    test_matches_val = np.isin(test_fingerprints, list(val_groups))

    model = sio.load(MODEL_PATH, trusted=sio.get_untrusted_types(file=MODEL_PATH))
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("El Pipeline congelado no ofrece predict_proba")
    x_test = test[columns]
    scores = model.predict_proba(x_test)[:, 1]
    y_true = test["label"].to_numpy()
    row_ids = test["id"].to_numpy()
    y_pred = (scores >= threshold).astype("int64")
    if not (len(scores) == len(y_true) == len(row_ids) == EXPECTED_TEST_ROWS):
        raise RuntimeError("La inferencia no produjo una predicción por fila")

    predictions = pd.DataFrame(
        {
            "row_id": row_ids,
            "y_true": y_true,
            "score_attack": scores,
            "y_pred": y_pred,
        }
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    fpr_denominator = fp + tn
    metrics = {
        "recall_attack": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_attack": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fpr": float(fp / fpr_denominator) if fpr_denominator else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    validation_reference = selection["validation_comparison"]["v2_selected_threshold"]
    validation_reference_metrics = {
        "recall_attack": validation_reference["recall_attack"],
        "precision_attack": validation_reference["precision_attack"],
        "fpr": validation_reference["fpr"],
        "macro_f1": validation_reference["macro_f1"],
    }
    deltas = {
        key: float(metrics[key] - validation_reference_metrics[key])
        for key in validation_reference_metrics
    }

    dvc_raw = yaml.safe_load((ROOT / "data/raw.dvc").read_text())["outs"][0]["md5"]
    dvc_processed = yaml.safe_load((ROOT / "data/processed.dvc").read_text())["outs"][0]["md5"]
    git_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()

    predictions.to_csv(PREDICTIONS_PATH, index=False, float_format="%.17g")
    result = {
        "version": selection["version_selected"],
        "model": {
            "artifact_sha256": model_sha,
            "huggingface_repo_id": hub_evidence["repo_id"],
            "huggingface_commit_sha": hub_revision,
            "frozen_git_tag": selection["git_reference"]["freeze_tag"],
            "frozen_git_commit": hub_model_git_commit,
            "mlflow_run_id": selection["run_id"],
        },
        "dataset": {
            "path": "data/raw/UNSW_NB15_testing-set.csv",
            "sha256": test_sha,
            "dvc_raw_directory_md5": dvc_raw,
            "dvc_processed_directory_md5": dvc_processed,
            "rows_evaluated": int(len(test)),
            "rows_expected": EXPECTED_TEST_ROWS,
            "all_rows_retained": len(predictions) == len(test),
            "class_counts": {str(k): int(v) for k, v in pd.Series(y_true).value_counts().sort_index().items()},
        },
        "decision": {
            "threshold": threshold,
            "rule": "y_pred = 1 (attack) si score_attack >= threshold; en otro caso y_pred = 0 (normal).",
            "positive_class": 1,
            "positive_class_meaning": "attack",
        },
        "confusion_matrix": {
            "labels": [0, 1],
            "rows_true_columns_predicted": [[int(matrix[0, 0]), int(matrix[0, 1])], [int(matrix[1, 0]), int(matrix[1, 1])]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "metrics": metrics,
        "zero_denominator_policy": "precision/recall/macro-F1 usan zero_division=0; FPR=0 si no hay negativos reales (FP+TN=0).",
        "validation_comparison": {
            "validation_reference_at_frozen_threshold": validation_reference_metrics,
            "test_minus_validation": deltas,
            "model_or_threshold_changed_after_freeze": False,
        },
        "input_overlap": {
            "method": "pandas hash_pandas_object uint64 sobre las 42 variables predictoras; se excluyen id/row_id, label y attack_cat; dtypes numéricos normalizados a float64 y categóricos a string.",
            "test_rows_matching_train": int(test_matches_train.sum()),
            "test_unique_groups_matching_train": int(len(train_overlap_groups)),
            "test_rows_matching_validation": int(test_matches_val.sum()),
            "test_unique_groups_matching_validation": int(len(val_overlap_groups)),
            "test_rows_matching_both_train_and_validation": int(np.logical_and(test_matches_train, test_matches_val).sum()),
            "test_unique_groups_matching_both": int(len(both_overlap_groups)),
            "test_unique_input_groups": int(len(test_groups)),
            "train_val_overlap_groups": len(train_groups & val_groups),
        },
        "evaluation": {
            "inference_calls": 1,
            "test_labels_used_for_selection_or_threshold": False,
            "test_rows_dropped": 0,
            "workspace_git_commit_at_evaluation": git_commit,
        },
        "predictions_file": "evidencias/predicciones-test.csv",
        "predictions_file_sha256": sha256(PREDICTIONS_PATH),
    }
    METRICS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"predictions={PREDICTIONS_PATH.relative_to(ROOT)}")
    print(f"metrics={METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
