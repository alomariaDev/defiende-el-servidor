# Defiende el servidor

Proyecto del taller de MLOps para clasificar tráfico normal/ataque con UNSW-NB15.

## Estado

Estructura inicial. El pipeline, las auditorías, los experimentos y las evidencias se completarán paso a paso con resultados obtenidos al ejecutar el código en este workspace.

## Datos

Los archivos fuente se conservan sin cambios en `data/raw/`. El objetivo es `label` (1 = ataque). Se excluyen `label`, `attack_cat` e `id` de las entradas del modelo. El conjunto de prueba se reserva para la evaluación final.

## Reproducibilidad

Configuraciones en `configs/`, código en `src/`, y evidencias en `evidencias/`. DVC, MLflow y Hugging Face se configurarán sin guardar credenciales en el repositorio.
