---
task_categories:
- tabular-classification
library_name: mlflow
---

# Defiende el servidor — v2

## Propósito

Clasificar filas de tráfico de red como normal (0) o ataque (1) usando un pipeline de scikit-learn con preprocesamiento y regresión logística. Este artefacto incluye el umbral de decisión de esta revisión.

## Datos

Entrenado con el conjunto de entrenamiento UNSW-NB15. El preprocesamiento se ajustó solo con el split de entrenamiento. La selección del umbral y las métricas aquí reportadas usan exclusivamente validación; no representan resultados en test. Se excluyen `label`, `attack_cat`, `id` y `row_id` de las entradas del modelo.

Versión DVC: `data/processed.dvc`, directorio `a065edaae55501c53239ceade3a3a8e0.dir`. Run MLflow: `ffdf524d57f749d28b5ed7fceb0fa0bf`. Commit Git de la versión: `f83efdcbb4eb75db01b4f27a79ed4a6393cc6e3c`. Commit Git del modelo fuente: `c2891e168109f7ab89dec90ce8d188ac687c1539`.

## Métricas de validación

Umbral de ataque: **0.3903680448122325**.

| Métrica | Valor |
|---|---:|
| Recall de ataque | 0.978925 |
| Precision de ataque | 0.930874 |
| FPR | 0.154911 |
| Macro-F1 | 0.924285 |
| Matriz de confusión (filas y columnas [0,1]) | `[[9465, 1735], [503, 23364]]` |

## Limitaciones

- Validación proviene de una partición de UNSW-NB15 y no garantiza rendimiento con tráfico operacional, otros periodos o distribuciones.
- El umbral de v2 se eligió sobre validación; puede sobreajustarse a esa partición.
- Las etiquetas y categorías del dataset pueden no cubrir ataques nuevos.
- No utilizar como único control de seguridad ni como sustituto de análisis de incidentes.

## Uso

Instala dependencias con `pip install -r requirements.txt`. Descarga esta revisión por su SHA completo y carga el pipeline MLflow:

```python
import json
from pathlib import Path
import mlflow.sklearn
from huggingface_hub import snapshot_download

repo_id = "mariisabell/unsw-nb15-detector"
revision = "<FULL_HUB_COMMIT_SHA>"
local = Path(snapshot_download(repo_id=repo_id, revision=revision))
model = mlflow.sklearn.load_model(str(local / "model"))
config = json.loads((local / "config.json").read_text())
score_attack = model.predict_proba(features)[:, 1]
y_pred = (score_attack >= config["decision_threshold"]).astype(int)
```

`features` debe ser un DataFrame con las columnas predictoras usadas en entrenamiento; excluye `label`, `attack_cat`, `id` y `row_id`.
