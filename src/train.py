"""Run configured validation experiments and log complete pipelines to MLflow."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pipeline(family: str, params: dict, seed: int, x_train: pd.DataFrame) -> Pipeline:
    categorical = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in x_train.columns if column not in categorical]
    numeric_steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline(categorical_steps), categorical),
        ],
        remainder="drop",
    )
    if family == "logistic_regression":
        estimator = LogisticRegression(random_state=seed, **params)
    elif family == "random_forest":
        estimator = RandomForestClassifier(random_state=seed, **params)
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])


def select_threshold_macro_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose the validation threshold maximizing macro-F1 in O(n log n)."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = y_true[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    thresholds = np.r_[sorted_scores[starts], np.nextafter(sorted_scores[-1], np.inf)]
    candidate_starts = np.r_[starts, len(y_true)]
    positives = int(np.sum(y_true == 1))
    negatives = len(y_true) - positives
    cumulative_positives = np.cumsum(sorted_labels)
    excluded_positives = np.zeros(len(candidate_starts), dtype=int)
    excluded_positives[1:] = cumulative_positives[candidate_starts[1:] - 1]
    tp = positives - excluded_positives
    predicted_positive = len(y_true) - candidate_starts
    fp = predicted_positive - tp
    fn = positives - tp
    tn = negatives - fp
    positive_denominator = 2 * tp + fp + fn
    negative_denominator = 2 * tn + fp + fn
    f1_positive = np.divide(2 * tp, positive_denominator, out=np.zeros_like(tp, dtype=float), where=positive_denominator != 0)
    f1_negative = np.divide(2 * tn, negative_denominator, out=np.zeros_like(tn, dtype=float), where=negative_denominator != 0)
    macro_f1 = (f1_positive + f1_negative) / 2
    recall = np.divide(tp, positives, out=np.zeros_like(tp, dtype=float), where=positives != 0)
    best = max(range(len(thresholds)), key=lambda i: (macro_f1[i], recall[i], -thresholds[i]))
    return float(thresholds[best])


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "confusion_matrix": [[int(matrix[0, 0]), int(matrix[0, 1])], [int(matrix[1, 0]), int(matrix[1, 1])]],
    }


def main() -> None:
    experiments = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
    data_config = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    train_path = ROOT / "data/processed/train.csv"
    val_path = ROOT / "data/processed/val.csv"
    train = pd.read_csv(train_path)
    validation = pd.read_csv(val_path)
    excluded = {"row_id", data_config["label_column"], *data_config["excluded_feature_columns"]}
    feature_columns = [column for column in train.columns if column not in excluded]
    x_train, y_train = train[feature_columns], train[data_config["label_column"]]
    x_val, y_val = validation[feature_columns], validation[data_config["label_column"]].to_numpy()

    tracking_dir = ROOT / "evidencias/mlflow"
    artifact_dir = tracking_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    database_path = tracking_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{database_path}")
    mlflow.set_experiment(
        experiments["experiment_name"],
        artifact_location=artifact_dir.as_uri(),
    )
    git_commit = os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip() or "uncommitted"
    data_version = {
        "training_csv_sha256": sha256(ROOT / data_config["train_path"]),
        "train_split_sha256": sha256(train_path),
        "validation_split_sha256": sha256(val_path),
    }
    results = []

    for item in experiments["configs"]:
        model = build_pipeline(item["family"], item["parameters"], experiments["seed"], x_train)
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_val)[:, 1]
        threshold = select_threshold_macro_f1(y_val, scores)
        chosen = metrics_at_threshold(y_val, scores, threshold)
        predictions = (scores >= chosen["threshold"]).astype(int)
        signature = infer_signature(x_val.head(10), model.predict(x_val.head(10)))

        with mlflow.start_run(run_name=item["id"]) as run:
            mlflow.set_tags({
                "config_id": item["id"],
                "family": item["family"],
                "hypothesis": item["hypothesis"],
                "git_commit": git_commit,
                "data_version": json.dumps(data_version, sort_keys=True),
                "target_positive_class": "1=attack",
            })
            mlflow.log_params({"seed": experiments["seed"], **item["parameters"]})
            mlflow.log_metrics({key: value for key, value in chosen.items() if isinstance(value, (int, float))})
            mlflow.log_dict(data_version, "data_version.json")
            mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=x_val.head(3))

        result = {
            "config_id": item["id"],
            "family": item["family"],
            "hypothesis": item["hypothesis"],
            "parameters": json.dumps(item["parameters"], sort_keys=True),
            "seed": experiments["seed"],
            "git_commit": git_commit,
            "data_version": json.dumps(data_version, sort_keys=True),
            "mlflow_run_id": run.info.run_id,
            "validation_rows": len(validation),
            **chosen,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    output = ROOT / "evidencias/experimentos.csv"
    pd.DataFrame(results).to_csv(output, index=False)
    print(f"Resultados guardados en {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
