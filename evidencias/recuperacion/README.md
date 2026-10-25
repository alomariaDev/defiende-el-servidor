# Fase 5: recuperación limpia

Esta prueba tiene dos referencias distintas:

- `v2` congelado para el modelo: `f83efdcbb4eb75db01b4f27a79ed4a6393cc6e3c`.
- El commit actual de `main` contiene este verificador y las referencias SHA del Hub. Se usa como **commit de recuperación** para clonar el harness; no cambia el tag ni el contenido congelado de `v2`.

## Preparar las predicciones de referencia

La carpeta contiene una muestra fija de 25 filas de validación, sus predicciones previas a la recuperación para v1/v2 y `manifest.json` con los SHA-256 calculados desde los archivos del workspace. Estos ficheros se preparan una sola vez y se versionan junto con el proyecto.

Para regenerar explícitamente la referencia en el workspace original:

```bash
.venv/bin/python src/recover.py baseline
```

Ese modo solo usa `data/processed/val.csv` y los modelos locales del Hub. Calcula SHA-256 del CSV de test sobre sus bytes, sin parsearlo ni utilizar sus etiquetas.

## Ejecutar en un checkout y entorno nuevos

Desde el workspace original actualizado, establece las credenciales de DagsHub en variables de entorno, nunca en archivos. El acceso DVC usa `AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY`. Si el repo HF es privado, define `HF_TOKEN`; no hace falta para un repo público. No incluyas los valores en los comandos registrados.

En el remote S3 de DagsHub, usa el **mismo token DagsHub como ambos valores** (`AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY`). La documentación oficial indica esa configuración para el almacenamiento S3 compatible de DagsHub ([DagsHub Storage](https://dagshub.com/docs/feature_guide/dagshub_storage/)). En Bash puedes pedirlo una sola vez sin mostrarlo en pantalla:

```bash
read -r -s -p "Token de acceso DagsHub: " DAGSHUB_TOKEN; printf '\n'
export AWS_ACCESS_KEY_ID="$DAGSHUB_TOKEN"
export AWS_SECRET_ACCESS_KEY="$DAGSHUB_TOKEN"
unset DAGSHUB_TOKEN
# Solo si el repositorio de Hugging Face es privado:
read -r -s -p "Token Hugging Face (HF_TOKEN): " HF_TOKEN; printf '\n'; export HF_TOKEN
```

```bash
git pull origin main
RECOVERY_COMMIT="$(git rev-parse HEAD)"
bash scripts/recover_clean.sh /tmp/defiende-recovery-$(date -u +%Y%m%dT%H%M%SZ) "$RECOVERY_COMMIT"
```

El script rechaza una carpeta de destino ya existente, clona Git exactamente en ese SHA, crea un entorno virtual nuevo, instala `requirements-recovery.txt` con `--no-cache-dir`, ejecuta `dvc pull` de originales/preparados y corre la verificación. El caché de Hugging Face queda aislado dentro de la carpeta de esa ejecución.

Después de completar la prueba, limpia las variables del shell:

```bash
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY HF_TOKEN
```

Para suministrar credenciales sin guardarlas en archivos, exporta las variables desde tu terminal antes de ejecutar el script. No pegues tokens en este repositorio ni en el chat. Si quieres escribir un secreto sin que se muestre en pantalla, puedes usar `read -r -s` y luego `export`; al terminar, ejecuta `unset` para retirarlo del entorno.

## Evidencias producidas

La ejecución crea `evidencias/recuperacion/run-<UTC>/` dentro del clon nuevo, con:

- `commands.log`: comandos ejecutados y sus salidas, sin valores de credenciales.
- `commands.log.sha256`: hash de ese log.
- `predicciones-recuperadas-v1.csv` y `predicciones-recuperadas-v2.csv` con `version,row_id,y_true,score_attack,y_pred`.
- `comparacion.json`: igualdad de filas/etiquetas/predicciones y diferencia absoluta máxima entre probabilidades.
- `hashes.json`: SHA-256 de los datos, modelos, entradas y salidas.
- `hf-cache/`: descargas realizadas en esta ejecución aislada.

La recuperación falla si el checkout, los punteros DVC, los datos, la muestra o los modelos no coinciden con el manifiesto. El archivo `UNSW_NB15_testing-set.csv` se hashea como bytes para verificar integridad; no se carga para inferencia ni se leen sus etiquetas.
