# Entrega: Defiende el servidor

## Fase 1: auditoría de datos

La auditoría se ejecutó sobre `data/raw/UNSW_NB15_training-set.csv`; el conjunto de test no se leyó para esta fase.

- Filas: 175.341; columnas: 45.
- SHA-256: `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`.
- Faltantes: 0.
- Clases: label 0 = 56.000; label 1 = 119.341.
- Entradas consideradas: 42; `id`, `attack_cat` y `label` excluidos.
- Grupos por huella de entradas: 9.748 grupos duplicados; 74.301 filas adicionales en esos grupos; grupo mayor de 415 filas.
- Grupos de huellas con etiquetas contradictorias: 229.

Política: conservar todas las filas y sus etiquetas; mantener cada huella de entradas en una sola partición; informar etiquetas contradictorias sin resolverlas ni borrar filas. `evidencias/auditoria.json` contiene tipos, faltantes por columna, conteo de clases y detalles de la huella.

## Split train/val

Split estratificado por grupos desde el archivo de entrenamiento, con semilla 42 y validación solicitada de 0,2:

- Train: 140.274 filas; label 0 = 44.800, label 1 = 95.474.
- Validación: 35.067 filas; label 0 = 11.200, label 1 = 23.867.
- Huellas compartidas entre particiones: 0.
- `data/processed/split_indices.csv` guarda la posición original, desde cero, y `row_id` cuando está disponible.

El procedimiento reproducible está en `src/prepare.py`; el reporte del split está en `evidencias/split.json`.

## Versionado DVC

Remote por defecto: `dagshub` en `s3://dvc`, con endpoint `https://dagshub.com/alomariaDev/defiende-el-servidor.s3`. Las credenciales se suministran en variables de entorno, no en archivos del proyecto.

- Datos originales: `data/raw.dvc`, huella de directorio `fe93051ae209ffb0128a31e7a478c009.dir`.
- Datos preparados: `data/processed.dvc`, huella de directorio `a065edaae55501c53239ceade3a3a8e0.dir`.

## Experimentos, selección y versiones del modelo

Pendiente de completar después de cerrar la Fase 1 y ejecutar las cuatro configuraciones correctamente.

## Recuperación y evaluación final

Pendiente. El conjunto de prueba permanece reservado hasta cerrar la selección de modelo y umbral.
