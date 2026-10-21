# Defiende el servidor

Proyecto del taller de MLOps para clasificar tráfico normal/ataque con UNSW-NB15.

## Estado

Fase 1 completada: auditoría y split agrupado desde el CSV de entrenamiento. Fase 2: pipeline y cuatro experimentos configurados; las métricas finales se generan al ejecutar `src/train.py` y se guardan en `evidencias/experimentos.csv`.

## Datos

Los archivos fuente se conservan sin cambios en `data/raw/`. El objetivo es `label` (1 = ataque). Se excluyen `label`, `attack_cat` e `id` de las entradas del modelo. El conjunto de prueba se reserva para la evaluación final.

## Reproducibilidad

Configuraciones en `configs/`, código en `src/`, y evidencias en `evidencias/`. DVC, MLflow y Hugging Face se configurarán sin guardar credenciales en el repositorio.

## Fase 1: auditoría y split

```bash
.venv/bin/python src/audit.py data/raw/UNSW_NB15_training-set.csv
.venv/bin/python src/prepare.py data/raw/UNSW_NB15_training-set.csv
```

La auditoría guarda SHA-256, dimensiones, tipos, faltantes, clases y grupos de huellas de entradas en `evidencias/auditoria.json`. La preparación usa únicamente el CSV de entrenamiento, agrupa huellas idénticas y guarda las posiciones originales en `data/processed/split_indices.csv`.

## Versionado DVC

El remote por defecto es DagsHub (`dagshub`, configurado como S3 compatible). Para versionar fuentes y derivados:

```bash
.venv/bin/dvc add data/raw data/processed
git add data/.gitignore data/raw.dvc data/processed.dvc
git commit -m "data: version source and prepared datasets"
.venv/bin/dvc push -r dagshub
```

La autenticación de DVC usa `AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY` en el entorno de la terminal; no guardar credenciales en archivos.

## Fase 2: experimentos

`data/processed.dvc` versiona `train.csv`, `val.csv` e `split_indices.csv`; su hash de directorio se registra en cada run como versión DVC. El preprocesamiento (imputación, codificación y escalado) está dentro del pipeline de scikit-learn y se ajusta al llamar `fit` únicamente con `train.csv`. Se excluyen `row_id`, `id`, `label` y `attack_cat` de las entradas.

Las cuatro configuraciones usan la semilla 42 y los mismos splits: regresión logística con `C=1` (línea base), regresión logística con `C=0.1` (más regularización), árbol de decisión con profundidad 5 (cambio de familia) y árbol con profundidad 15 (mayor complejidad). Cada hipótesis y parámetros están en `configs/experiments.yaml`.

El umbral se selecciona maximizando macro-F1 en validación. La clase positiva es `1` (ataque). Para denominadores cero, precision, recall y F1 usan `zero_division=0`; FPR es 0 si no hay negativos reales. La matriz de confusión se guarda como artefacto JSON en cada run. Los modelos completos se registran en MLflow local; no se evalúa el conjunto de test durante esta fase.

Para ejecutar desde la raíz del proyecto:

```bash
.venv/bin/python src/train.py
```
