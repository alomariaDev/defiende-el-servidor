#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "Uso: bash scripts/recover_clean.sh <carpeta-nueva> <commit-recuperacion-SHA-completo>" >&2
  exit 2
fi

WORK_DIR="$(realpath -m "$1")"
RECOVERY_COMMIT="$2"
REPO_URL="https://github.com/alomariaDev/defiende-el-servidor.git"
if [[ ! "$RECOVERY_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "El commit debe ser un SHA Git completo de 40 caracteres." >&2
  exit 2
fi
if [[ -e "$WORK_DIR" ]]; then
  echo "La carpeta ya existe; usa una ruta nueva para no reutilizar cachés: $WORK_DIR" >&2
  exit 2
fi
mkdir -p "$WORK_DIR"
LOG_FILE="$WORK_DIR/commands.log"
exec > >(tee -a "$LOG_FILE") 2>&1

run() {
  printf '$'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

echo "Recovery checkout solicitado: $RECOVERY_COMMIT"
echo "Workspace limpio: $WORK_DIR"
for credential_name in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
  if [[ -z "${!credential_name:-}" ]]; then
    echo "Falta la variable de entorno $credential_name para el remote DagsHub." >&2
    exit 3
  fi
  echo "$credential_name=present (valor omitido)"
done
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN=present (valor omitido)"
else
  echo "HF_TOKEN=absent (solo se necesita si el repo HF es privado)"
fi

REPO_DIR="$WORK_DIR/repo"
VENV_DIR="$WORK_DIR/venv"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="$REPO_DIR/evidencias/recuperacion/run-$RUN_ID"

finalize() {
  status=$?
  if [[ -d "$REPO_DIR/evidencias/recuperacion" ]]; then
    mkdir -p "$OUTPUT_DIR"
    cp "$LOG_FILE" "$OUTPUT_DIR/commands.log"
    sha256sum "$OUTPUT_DIR/commands.log" > "$OUTPUT_DIR/commands.log.sha256"
    echo "Log de comandos: $OUTPUT_DIR/commands.log"
  fi
  return "$status"
}
trap finalize EXIT

run git clone --no-checkout "$REPO_URL" "$REPO_DIR"
run git -C "$REPO_DIR" checkout --detach "$RECOVERY_COMMIT"
run python3 -m venv "$VENV_DIR"
run "$VENV_DIR/bin/python" -m pip install --no-cache-dir -r "$REPO_DIR/requirements-recovery.txt"
run "$VENV_DIR/bin/dvc" -C "$REPO_DIR" config --local cache.dir "$WORK_DIR/dvc-cache"
run "$VENV_DIR/bin/dvc" -C "$REPO_DIR" pull data/raw.dvc data/processed.dvc

export HF_HOME="$WORK_DIR/hf-home"
echo "HF_HOME=$HF_HOME (directorio nuevo; el valor del token nunca se registra)"
run "$VENV_DIR/bin/python" "$REPO_DIR/src/recover.py" verify \
  --output-dir "$OUTPUT_DIR" \
  --expected-commit "$RECOVERY_COMMIT"
echo "Evidencias de recuperación: $OUTPUT_DIR"
