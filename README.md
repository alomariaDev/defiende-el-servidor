# Defiende el servidor

Clasificador binario de tráfico normal (0) y ataque (1) con UNSW-NB15. El proyecto usa Python, scikit-learn, DVC con almacenamiento remoto DagsHub, MLflow local y Hugging Face Hub. Los datos fuente están bajo DVC y no se modifican.

## Requisitos e instalación

La ejecución registrada se hizo con Python 3.14.4 y CPU. Desde la raíz del repositorio, crea un entorno e instala las versiones fijadas:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Recuperar los datos

Los punteros DVC son `data/raw.dvc` y `data/processed.dvc`; los archivos fuente y los splits se descargan del remote DagsHub. Define las credenciales solo en la sesión de terminal. DagsHub S3 usa el token como ambas claves AWS:

```bash
read -r -s -p "Token DagsHub: " DAGSHUB_TOKEN; printf '\n'
export AWS_ACCESS_KEY_ID="$DAGSHUB_TOKEN"
export AWS_SECRET_ACCESS_KEY="$DAGSHUB_TOKEN"
unset DAGSHUB_TOKEN
dvc pull data/raw.dvc data/processed.dvc
```

Para un clone nuevo también puedes comprobar los objetos con `dvc status -c`. Los hashes DVC esperados están en `data/*.dvc`; SHA-256 de los CSV recuperados y verificados están en `evidencias/recuperacion/`.

### Recuperar desde el ZIP de backup DVC

El archivo `dvc-backup.zip` se distribuye aparte del repositorio porque contiene una copia local del cache DVC y no está guardado en Git. Coloca el ZIP en la carpeta padre del clon del repositorio. Desde la raíz del repo, descomprímelo y configura el remote local:

```bash
python -m zipfile -e ../dvc-backup.zip ..
dvc remote add --local backup ../dvc-backup
dvc pull -r backup data/raw.dvc data/processed.dvc
```

La configuración se guarda solo en `.dvc/config.local`. Si el ZIP está en otra ubicación, ajusta la ruta del archivo al extraerlo y la ruta del directorio en `dvc remote add`.

## Ejecutar un experimento

`configs/experiments.yaml` declara las cuatro configuraciones y sus hipótesis. El script reutiliza los splits ya preparados, ajusta la imputación, codificación, escalado y estimador solo con train, evalúa umbrales solo en validación y registra el pipeline completo localmente:

```bash
python src/train.py
```

MLflow queda en `evidencias/mlflow/`; el resumen tabular queda en `evidencias/experimentos.csv`. Esta instrucción vuelve a entrenar y crea runs adicionales; no hace falta para recuperar los resultados ya entregados.

## Descargar y predecir v1 o v2

Las revisiones publicadas están en [`mariisabell/unsw-nb15-detector`](https://huggingface.co/mariisabell/unsw-nb15-detector). Estos comandos descargan cada revisión por SHA completo y predicen las primeras 25 filas de validación. Con un repo privado, define `HF_TOKEN` en el entorno antes de ejecutar; no lo pongas en el código.

```bash
read -r -s -p "Token Hugging Face (solo si el repo es privado): " HF_TOKEN; printf '\n'
export HF_TOKEN
python - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd
import skops.io as sio
from huggingface_hub import snapshot_download

root = Path('.')
evidence = json.loads((root / 'evidencias/huggingface.json').read_text())
sample = pd.read_csv(root / 'data/processed/val.csv').head(25)
features = sample.drop(columns=['row_id', 'id', 'label', 'attack_cat'], errors='ignore')

for version in ('v1', 'v2'):
    revision = evidence['revisions'][version]['commit_sha']
    threshold = evidence['revisions'][version]['threshold']
    folder = Path(snapshot_download(
        repo_id=evidence['repo_id'], repo_type='model', revision=revision,
        token=os.environ.get('HF_TOKEN') or None,
    ))
    model_path = folder / 'model/model.skops'
    model = sio.load(model_path, trusted=sio.get_untrusted_types(file=model_path))
    scores = model.predict_proba(features)[:, 1]
    predictions = (scores >= threshold).astype(int)
    print(version, revision, 'threshold=', threshold)
    print(pd.DataFrame({'row_id': sample['row_id'], 'score_attack': scores,
                        'y_pred': predictions}).to_string(index=False))
PY
```

El umbral está en el `config.json` correspondiente y se registra también en `evidencias/huggingface.json`: v1 usa 0.5; v2 usa 0.3903680448122325. La regla es ataque cuando `score_attack >= threshold`. El Pipeline incluye su propio preprocesamiento.

## Recuperación reproducible

La prueba documentada usó un checkout y cachés nuevos; el script exige un directorio de salida inexistente. Para repetirla desde un clone nuevo, primero instala `requirements-recovery.txt`, recupera DVC como arriba y usa el commit de recuperación documentado:

```bash
python -m pip install -r requirements-recovery.txt
git checkout ec8b14923f3a5539250030045b47793c8637d4f8
python src/recover.py verify \
  --expected-commit ec8b14923f3a5539250030045b47793c8637d4f8 \
  --output-dir evidencias/recuperacion/run-nueva
```

La salida incluye predicciones v1/v2, hashes y comparación contra el baseline de validación. La recuperación que ya se ejecutó y sus logs están en `evidencias/recuperacion/run-20261025T195751Z/`.

## Evaluación final y documentos

La evaluación del test completo ya se ejecutó una vez con v2 congelada. `src/evaluate_test.py` se protege contra una segunda ejecución cuando ya existen sus artefactos. No lo vuelvas a correr para reproducir cifras: consulta `evidencias/predicciones-test.csv` y `evidencias/metricas-test.json`.

- [Ficha de entrega](entrega.md)
- [Borrador del informe](informe.md)
- [Configuración de datos](configs/data.yaml) y [experimentos](configs/experiments.yaml)
- Evidencias: `evidencias/`, incluidos auditoría, splits, runs MLflow, selección, recuperación y evaluación.

## Declaración de uso de IA

Se utilizó asistencia de IA (Codex) para guiar la implementación, revisar inconsistencias y redactar documentación. Las métricas, hashes, identificadores de ejecución y referencias que se reportan se transcribieron de los archivos y comandos del proyecto; no se generaron resultados sintéticos. La autora revisa y asume la responsabilidad del contenido final.
