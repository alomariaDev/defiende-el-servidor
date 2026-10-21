"""Publish model revisions to Hugging Face Hub and verify by immutable SHA."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import skops.io as skops
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repo, e.g. user/model")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN no está definido en el entorno; no se leyó ningún archivo de credenciales.")

    selection_path = ROOT / "evidencias/seleccion.json"
    selection = json.loads(selection_path.read_text())
    expected_model_sha = selection["model"]["artifact_sha256"]
    revisions: dict[str, dict] = {}
    api = HfApi(token=token)
    api.repo_info(repo_id=args.repo_id, repo_type="model")

    for version in ("v1", "v2"):
        package_dir = ROOT / "evidencias/huggingface" / version
        config_path = package_dir / "config.json"
        config = json.loads(config_path.read_text())
        if config.get("repo_id") != args.repo_id:
            config["repo_id"] = args.repo_id
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        # MLflow exports can contain a machine-local artifact_path.
        # Remove it so the downloaded model resolves model.skops relative to itself.
        mlmodel_path = package_dir / "model/MLmodel"
        mlmodel_text = mlmodel_path.read_text()
        mlmodel_text = "\n".join(
            line for line in mlmodel_text.splitlines()
            if not line.startswith("artifact_path:")
        ) + "\n"
        mlmodel_path.write_text(mlmodel_text)
        card_path = package_dir / "README.md"
        card_path.write_text(card_path.read_text().replace("<HF_REPO_ID>", args.repo_id))
        model_path = package_dir / "model/model.skops"
        actual_sha = sha256(model_path)
        if actual_sha != expected_model_sha:
            raise RuntimeError(f"SHA-256 inesperado para {version}: {actual_sha}")

        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(package_dir),
            commit_message=f"Publish {version} model revision",
        )
        commit_sha = commit.oid
        if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
            raise RuntimeError(f"El Hub no devolvió un SHA Git completo: {commit_sha}")
        revisions[version] = {
            "commit_sha": commit_sha,
            "run_id": config["mlflow_run_id"],
            "threshold": config["decision_threshold"],
            "model_sha256": actual_sha,
            "git_tag": config["git_tag"],
            "git_commit": config["git_commit"],
            "dvc_pointer": config["dvc_pointer"],
            "dvc_directory_md5": config["dvc_directory_md5"],
        }
        print(f"{version}_commit_sha={commit_sha}", flush=True)

    if revisions["v1"]["commit_sha"] == revisions["v2"]["commit_sha"]:
        raise RuntimeError("v1 y v2 no quedaron en commits distintos.")

    for version in ("v1", "v2"):
        commit_sha = revisions[version]["commit_sha"]
        downloaded = Path(snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            revision=commit_sha,
            token=token,
        ))
        model_file = downloaded / "model/model.skops"
        trusted_types = skops.get_untrusted_types(file=model_file)
        loaded = skops.load(model_file, trusted=trusted_types)
        if not hasattr(loaded, "predict_proba"):
            raise RuntimeError(f"El modelo descargado para {version} no expone predict_proba.")
        revisions[version]["downloaded_by_sha"] = commit_sha
        revisions[version]["loaded_pipeline_steps"] = list(loaded.named_steps)
        print(f"{version}_download_load=OK revision={commit_sha} steps={list(loaded.named_steps)}", flush=True)

    evidence = {
        "repo_id": args.repo_id,
        "repo_type": "model",
        "revisions": revisions,
        "verification": "Each revision was downloaded by its full commit SHA and loaded from the skops artifact.",
    }
    evidence_path = ROOT / "evidencias/huggingface.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    selection["huggingface"] = {
        "repo_id": args.repo_id,
        "v1_commit_sha": revisions["v1"]["commit_sha"],
        "v2_commit_sha": revisions["v2"]["commit_sha"],
    }
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    print(f"evidence={evidence_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
