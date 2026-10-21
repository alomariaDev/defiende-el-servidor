# Defiende el servidor

Proyecto del taller de MLOps para clasificar tráfico normal/ataque con UNSW-NB15.

## Estado

Estructura inicial. El pipeline, las auditorías, los experimentos y las evidencias se completarán paso a paso con resultados obtenidos al ejecutar el código en este workspace.

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
