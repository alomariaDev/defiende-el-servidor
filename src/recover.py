"""Create a validation-only recovery baseline or verify it in a clean clone."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import skops.io as sio
import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
RECOVERY_DIR = ROOT / "evidencias/recuperacion"
RAW_FILES = [
    "data/raw/UNSW_NB15_training-set.csv",
    "data/raw/UNSW_NB15_testing-set.csv",
]
PREPARED_FILES = [
    "data/processed/train.csv",
    "data/processed/val.csv",
    "data/processed/split_indices.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_features(frame: pd.DataFrame) -> pd.DataFrame:
    excluded = {"id", "row_id", "label", "attack_cat"}
    return frame[[column for column in frame.columns if column.lower() not in excluded]]


def predict(model, frame: pd.DataFrame, threshold: float, version: str) -> pd.DataFrame:
    scores = model.predict_proba(model_features(frame))[:, 1]
    return pd.DataFrame(
        {
            "version": version,
            "row_id": frame["row_id"].to_numpy(),
            "y_true": frame["label"].to_numpy(),
            "score_attack": scores,
            "y_pred": (scores >= threshold).astype("int64"),
        }
    )


def get_refs() -> dict:
    evidence = json.loads((ROOT / "evidencias/huggingface.json").read_text())
    selection = json.loads((ROOT / "evidencias/seleccion.json").read_text())
    refs = {}
    for version in ("v1", "v2"):
        hub = evidence["revisions"][version]
        config = json.loads((ROOT / "evidencias/huggingface" / version / "config.json").read_text())
        refs[version] = {
            "repo_id": evidence["repo_id"],
            "commit_sha": hub["commit_sha"],
            "threshold": float(config["decision_threshold"]),
            "model_sha256": hub["model_sha256"],
            "git_tag": hub["git_tag"],
            "git_commit": hub["git_commit"],
            "mlflow_run_id": hub["run_id"],
        }
    if selection.get("huggingface", {}).get("v1_commit_sha") != refs["v1"]["commit_sha"]:
        raise RuntimeError("SHA v1 no coincide entre seleccion.json y huggingface.json")
    if selection.get("huggingface", {}).get("v2_commit_sha") != refs["v2"]["commit_sha"]:
        raise RuntimeError("SHA v2 no coincide entre seleccion.json y huggingface.json")
    return refs


def dvc_md5(pointer: Path) -> str:
    metadata = yaml.safe_load(pointer.read_text())
    return str(metadata["outs"][0]["md5"])


def baseline() -> None:
    """Freeze a deterministic 25-row validation input and its pre-recovery outputs."""
    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(ROOT / "data/processed/val.csv")
    if "row_id" not in val or "label" not in val:
        raise RuntimeError("val.csv debe incluir row_id y label")
    sample = val.head(25).copy()
    if len(sample) < 20:
        raise RuntimeError(f"Se requieren al menos 20 filas; hay {len(sample)}")
    sample_path = RECOVERY_DIR / "muestra-validacion.csv"
    sample.to_csv(sample_path, index=False, float_format="%.17g")

    refs = get_refs()
    baseline_files = {}
    model_hashes = {}
    for version in ("v1", "v2"):
        model_path = ROOT / "evidencias/huggingface" / version / "model/model.skops"
        model_hash = sha256(model_path)
        if model_hash != refs[version]["model_sha256"]:
            raise RuntimeError(f"SHA del modelo local no coincide con Hugging Face ({version})")
        model = sio.load(model_path, trusted=sio.get_untrusted_types(file=model_path))
        predictions = predict(model, sample, refs[version]["threshold"], version)
        output_path = RECOVERY_DIR / f"predicciones-base-{version}.csv"
        predictions.to_csv(output_path, index=False, float_format="%.17g")
        baseline_files[version] = {
            "path": str(output_path.relative_to(ROOT)),
            "sha256": sha256(output_path),
        }
        model_hashes[version] = model_hash

    data_hashes = {name: sha256(ROOT / name) for name in RAW_FILES + PREPARED_FILES}
    manifest = {
        "purpose": "Baseline fija pre-recuperación; la muestra sale solo de val.csv.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_sample": {
            "path": str(sample_path.relative_to(ROOT)),
            "rows": int(len(sample)),
            "row_ids": [int(value) for value in sample["row_id"].tolist()],
            "sha256": sha256(sample_path),
            "validation_source": "data/processed/val.csv; primeras 25 filas en orden almacenado",
        },
        "data_sha256": data_hashes,
        "dvc": {
            "raw_pointer": "data/raw.dvc",
            "raw_directory_md5": dvc_md5(ROOT / "data/raw.dvc"),
            "processed_pointer": "data/processed.dvc",
            "processed_directory_md5": dvc_md5(ROOT / "data/processed.dvc"),
        },
        "versions": refs,
        "local_model_sha256": model_hashes,
        "baseline_predictions": baseline_files,
        "test_policy": "UNSW_NB15_testing-set.csv solo se hashea como bytes; no se parsea ni se accede a sus etiquetas.",
    }
    manifest_path = RECOVERY_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path.relative_to(ROOT)),
        "sample_rows": len(sample),
        "row_ids": manifest["validation_sample"]["row_ids"],
        "data_sha256": data_hashes,
        "baseline_predictions": baseline_files,
        "model_sha256": model_hashes,
    }, ensure_ascii=False, indent=2))


def verify(output_dir: Path, expected_commit: str) -> None:
    manifest_path = RECOVERY_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if len(expected_commit) != 40 or any(c not in "0123456789abcdef" for c in expected_commit):
        raise ValueError("El commit de recuperación debe ser un SHA completo de 40 caracteres")
    actual_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != expected_commit:
        raise RuntimeError(f"Checkout no coincide: esperado {expected_commit}, actual {actual_commit}")
    if output_dir.exists():
        raise FileExistsError(f"La carpeta de salida ya existe; no se reutilizan resultados: {output_dir}")
    output_dir.mkdir(parents=True)

    pointer_hashes = {
        "raw_directory_md5": dvc_md5(ROOT / manifest["dvc"]["raw_pointer"]),
        "processed_directory_md5": dvc_md5(ROOT / manifest["dvc"]["processed_pointer"]),
    }
    for key, actual in pointer_hashes.items():
        if actual != manifest["dvc"][key]:
            raise RuntimeError(f"Hash DVC {key} distinto: {actual}")

    actual_data_hashes = {}
    for relative_path, expected in manifest["data_sha256"].items():
        actual = sha256(ROOT / relative_path)
        actual_data_hashes[relative_path] = actual
        if actual != expected:
            raise RuntimeError(f"SHA-256 distinto para {relative_path}: {actual}")

    sample_path = ROOT / manifest["validation_sample"]["path"]
    if sha256(sample_path) != manifest["validation_sample"]["sha256"]:
        raise RuntimeError("La muestra fija de validación no coincide con su SHA-256")
    sample = pd.read_csv(sample_path)
    validation = pd.read_csv(ROOT / "data/processed/val.csv")
    if validation["row_id"].duplicated().any():
        raise RuntimeError("row_id duplicado en val.csv; no se puede reconstruir la muestra sin ambigüedad")
    if sample["row_id"].tolist() != manifest["validation_sample"]["row_ids"]:
        raise RuntimeError("Los row_id de la muestra cambiaron")
    by_id = validation.set_index("row_id", drop=False)
    selected = by_id.loc[manifest["validation_sample"]["row_ids"], sample.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(selected, sample, check_dtype=False, check_exact=True)

    refs = manifest["versions"]
    output_hashes = {}
    comparisons = {}
    model_hashes = {}
    for version in ("v1", "v2"):
        ref = refs[version]
        root = Path(snapshot_download(
            repo_id=ref["repo_id"],
            repo_type="model",
            revision=ref["commit_sha"],
            cache_dir=str(output_dir / "hf-cache"),
            token=os.environ.get("HF_TOKEN") or None,
        ))
        model_path = root / "model/model.skops"
        model_hash = sha256(model_path)
        if model_hash != ref["model_sha256"]:
            raise RuntimeError(f"SHA del artefacto descargado no coincide ({version}): {model_hash}")
        model_hashes[version] = model_hash
        model = sio.load(model_path, trusted=sio.get_untrusted_types(file=model_path))
        if not hasattr(model, "predict_proba"):
            raise RuntimeError(f"El modelo {version} no expone predict_proba")
        result = predict(model, sample, ref["threshold"], version)
        output_path = output_dir / f"predicciones-recuperadas-{version}.csv"
        result.to_csv(output_path, index=False, float_format="%.17g")
        output_hashes[str(output_path.relative_to(ROOT))] = sha256(output_path)

        baseline_path = ROOT / manifest["baseline_predictions"][version]["path"]
        baseline_hash = sha256(baseline_path)
        if baseline_hash != manifest["baseline_predictions"][version]["sha256"]:
            raise RuntimeError(f"Baseline alterada para {version}: {baseline_hash}")
        before = pd.read_csv(baseline_path)
        if before[["version", "row_id", "y_true", "y_pred"]].to_dict("records") != result[["version", "row_id", "y_true", "y_pred"]].to_dict("records"):
            labels_equal = False
        else:
            labels_equal = True
        max_abs = float((before["score_attack"] - result["score_attack"]).abs().max())
        comparisons[version] = {
            "baseline_path": str(baseline_path.relative_to(ROOT)),
            "recovered_path": str(output_path.relative_to(ROOT)),
            "rows_compared": int(len(sample)),
            "row_ids_equal": before["row_id"].tolist() == result["row_id"].tolist(),
            "y_true_equal": before["y_true"].tolist() == result["y_true"].tolist(),
            "predictions_equal": labels_equal,
            "max_abs_score_difference": max_abs,
            "baseline_sha256": baseline_hash,
            "recovered_sha256": output_hashes[str(output_path.relative_to(ROOT))],
            "loaded_pipeline_steps": list(model.named_steps),
            "hub_revision": ref["commit_sha"],
        }

    result = {
        "status": "passed" if all(
            item["row_ids_equal"] and item["y_true_equal"] and item["predictions_equal"]
            for item in comparisons.values()
        ) else "failed_prediction_comparison",
        "recovery_checkout_commit": actual_commit,
        "frozen_model_git_commit_v2": refs["v2"]["git_commit"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dvc_pointer_hashes": pointer_hashes,
        "verified_data_sha256": actual_data_hashes,
        "downloaded_model_sha256": model_hashes,
        "versions": comparisons,
        "test_labels_used": False,
        "test_file_sha256_only": actual_data_hashes["data/raw/UNSW_NB15_testing-set.csv"],
    }
    (output_dir / "comparacion.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "hashes.json").write_text(json.dumps({
        "manifest_sha256": sha256(manifest_path),
        "validation_sample_sha256": sha256(sample_path),
        "verified_data_sha256": actual_data_hashes,
        "downloaded_model_sha256": model_hashes,
        "recovered_prediction_sha256": output_hashes,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "verify"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    if args.mode == "baseline":
        baseline()
    else:
        if args.output_dir is None or args.expected_commit is None:
            parser.error("verify requiere --output-dir y --expected-commit")
        verify(args.output_dir.resolve(), args.expected_commit)


if __name__ == "__main__":
    main()
