# Entrega — Defiende el servidor

## Identificación y repositorios

- Autora: Maria Isabell Guzman
- Correo Git: `182182664+alomariaDev@users.noreply.github.com`
- GitHub: <https://github.com/alomariaDev/defiende-el-servidor>
- Remote DVC: DagsHub HTTPS `https://dagshub.com/alomariaDev/defiende-el-servidor.dvc`; también está configurado el remote S3 `s3://dvc` con endpoint `https://dagshub.com/alomariaDev/defiende-el-servidor.s3`. Son almacenes DVC separados. La prueba de recuperación autenticada del remote HTTPS y el estado de lectura anónima están documentados abajo en “Acceso a DVC desde DagsHub”.
- Hugging Face: <https://huggingface.co/mariisabell/unsw-nb15-detector>
- Entorno registrado: Python 3.14.4, CPU. Paquetes directos fijados en `requirements.txt` (incluye scikit-learn 1.9.1, MLflow 3.17.2, DVC 3.67.1 y huggingface_hub 2.2.0).

## Datos, auditoría y partición

Fuente: UNSW-NB15. El objetivo es `label`, con 1 como ataque y clase positiva. `label`, `attack_cat`, `id` y el identificador derivado `row_id` no son entradas del modelo. Los CSV originales permanecen sin modificar bajo DVC.

| Archivo/conjunto | Filas | SHA-256 |
|---|---:|---|
| Training original | 175.341 | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` |
| Testing original | 82.332 | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` |
| train.csv preparado | 140.274 | `a3e8529f58a72c7ad07251e0e9edced1714b8610c25bb7cea3294fdb5e16cd0f` |
| val.csv preparado | 35.067 | `6dc01d121473736aa962d9289e1e5a93ca4396a25075a32cd2f5e86211c36e2b` |
| índices de split | — | `3aba38d80b2125c9e6c1eb6c4ede8bad59c526cfaba96ad8075414d11a94863f` |

Auditoría del training: 45 columnas, 42 predictoras, 0 faltantes; label 0: 56.000 y label 1: 119.341. Se detectaron 9.748 grupos de predictores repetidos (74.301 filas adicionales sobre una primera ocurrencia), tamaño máximo de grupo 415, y 229 grupos con etiquetas contradictorias. Política: conservar todas las filas/etiquetas, asignar cada huella idéntica a una sola partición e informar las contradicciones sin resolverlas. Split desde training solamente, `StratifiedGroupKFold` primer fold, semilla 42, validación solicitada 20%; train label 0/1: 44.800/95.474; val: 11.200/23.867; grupos compartidos train-val: 0. Los índices apuntan a posiciones originales desde cero.

Hashes DVC de directorio: originales `fe93051ae209ffb0128a31e7a478c009.dir`; preparados/splits `a065edaae55501c53239ceade3a3a8e0.dir`. Punteros: `data/raw.dvc` y `data/processed.dvc`.

## Pipeline y experimentos de validación

Los transformadores de numéricas (imputación por mediana y escalado estándar) y categóricas (imputación más frecuente y one-hot con categorías desconocidas ignoradas) forman parte del Pipeline scikit-learn con el estimador. Se ajusta mediante `fit` únicamente con train. Las cuatro configuraciones comparten split y semilla 42. Umbral por configuración: máximo macro-F1 sobre scores de validación. FPR se calcula como FP/(FP+TN), o 0 si no hay negativos reales. Precision/recall/F1 usan `zero_division=0`.

| Configuración / hipótesis resumida | Parámetros principales | Run ID MLflow | Umbral | Recall ataque | Precision ataque | FPR | Macro-F1 | Matriz [[TN,FP],[FN,TP]] |
|---|---|---|---:|---:|---:|---:|---:|---|
| `lr_c1`: referencia lineal | C=1, balanceado, lbfgs | `ffdf524d57f749d28b5ed7fceb0fa0bf` | 0.3903680448122325 | 0.9789248753509029 | 0.9308737399896411 | 0.1549107142857143 | 0.9242845966144494 | [[9465,1735],[503,23364]] |
| `lr_c0_1`: más regularización podría generalizar mejor | C=0.1, balanceado, lbfgs | `22f34c10393e4202a836abe8c67a5147` | 0.40173156466464144 | 0.9798047513302887 | 0.9286395044079104 | 0.16044642857142857 | 0.9227249711857688 | [[9403,1797],[482,23385]] |
| `dt_depth5`: capturar no linealidad controlando complejidad | profundidad 5, balanceado | `aacb634fbbae4a04b35ef9d23dfb77f7` | 0.5302063907328993 | 0.9401265345456069 | 0.9565180322278114 | 0.09107142857142857 | 0.9204407774366041 | [[10180,1020],[1429,22438]] |
| `dt_depth15`: más interacciones, mayor riesgo de sobreajuste | profundidad 15, balanceado | `dfb5ea69ac0a4691a9762c94535dcbd7` | 0.3537730527816762 | 0.9558805044622282 | 0.9655084853358161 | 0.07276785714285715 | 0.9390771244547116 | [[10385,815],[1053,22814]] |

Los cuatro registros reportan código fuente en Git `c2891e168109f7ab89dec90ce8d188ac687c1539` y datos preparados DVC `a065edaae55501c53239ceade3a3a8e0.dir`. Run IDs, parámetros, hipótesis y resultados completos están en `evidencias/experimentos.csv` (SHA-256 `8d93dc0ba6557adfb6e52b879ccc692880a046721defda9ca4ef0e7595ec2030`) y las carpetas/artifacts MLflow en `evidencias/mlflow/`.

## Selección, versiones y publicación

- **v1**: tag Git `v1`, commit `c2891e168109f7ab89dec90ce8d188ac687c1539`; regresión logística C=1, umbral fijo 0.5. Validación: recall 0.9539531570788118; precision 0.9479162329822224; FPR 0.11169642857142857; macro-F1 0.9226488208398218; matriz [[9949,1251],[1099,22768]].
- **v2**: tag Git `v2`, commit de congelación `f83efdcbb4eb75db01b4f27a79ed4a6393cc6e3c`. Único cambio: umbral 0.5 → 0.3903680448122325; Pipeline y preprocesamiento sin cambios. Se eligió maximizando macro-F1 en validación para los scores de `lr_c1`; no se consultaron etiquetas de test para seleccionar modelo/umbral. Validación: recall 0.9789248753509029; precision 0.9308737399896411; FPR 0.1549107142857143; macro-F1 0.9242845966144494; matriz [[9465,1735],[503,23364]]. Frente a v1 reduce FN en 596 y aumenta FP en 484 en validación.
- Run MLflow asociado: `ffdf524d57f749d28b5ed7fceb0fa0bf`; model ID `m-021b714efe23458780e0273d739de6ad`; SHA-256 del artefacto `model.skops`: `e0f3019485ecfaa7418c7d4c31b7d130b1c5f003162de5f925d83561b7683dc3`.
- Commits completos de Hugging Face: v1 `04f723fe871e9b2f3b10b2ccd88cb2b45f587b53`; v2 `03e583e00e2c3d5dda0d0eb1fb9364ab1b86c2d6`. Ambas revisiones se descargaron por SHA completo y se cargaron como Pipeline. Referencia estructurada: `evidencias/huggingface.json`; model cards/configs: `evidencias/huggingface/v1/` y `v2/`.
- La selección congelada no es el mayor macro-F1 de todas las cuatro filas: `dt_depth15` alcanzó 0.9390771244547116. v2 conserva la regresión logística de referencia y cambia solo su umbral. Se documenta para que la decisión sea auditable; no se reabre tras el test.
- `evidencias/seleccion.json` SHA-256: `9ec109a888609ced6e182ab1322c08379a41474e198c3c243082a2665da9b48e`.

## Recuperación limpia

La prueba exitosa se ejecutó en checkout nuevo `ec8b14923f3a5539250030045b47793c8637d4f8`; verificó los SHA-256 de los dos datos originales y tres archivos preparados, descargó v1/v2 por SHA de Hub y cargó `preprocessor` y `estimator`. Sobre la misma muestra fija de 25 filas de validación coincidieron row_id, y_true y y_pred. Diferencia absoluta máxima de score para ambos modelos: `2.220446049250313e-16` (tolerancia `1e-12`).

Archivos de la ejecución: `evidencias/recuperacion/run-20261025T195751Z/commands.log`, `comparacion.json`, `hashes.json`, y `predicciones-recuperadas-v1.csv`/`v2.csv`; baseline y muestra en la raíz de `evidencias/recuperacion/`. SHA-256 del log `792671abbdd5df3d68548568c244db0cec7666bd7e925d91515a4a8de67488cc`; comparación `0a88b8e505fcfea48450ae63e229c5a1c736ae5c2a99a41f4f247f95e49d9c06`.

## Acceso a DVC desde DagsHub

La autora informa que el repositorio de DagsHub está configurado como público. En un clon limpio sin credenciales, `dvc pull -r dagshub-public data/raw.dvc data/processed.dvc` no pudo leer los objetos: faltan los hashes de directorio `fe93051ae209ffb0128a31e7a478c009.dir` y `a065edaae55501c53239ceade3a3a8e0.dir`; las solicitudes HTTP anónimas a ambos objetos devolvieron 404. El acceso S3 sin credenciales devolvió 401. Por tanto, esta entrega no afirma que el `dvc pull` anónimo haya funcionado.

En un clon temporal nuevo, `dvc fetch --verbose -r dagshub-public data/raw.dvc data/processed.dvc` solicitó autenticación básica de DagsHub y terminó con `7 files fetched`. Después, `dvc checkout` materializó los datos en ese clon y los SHA-256 de los dos CSV originales, `train.csv`, `val.csv` y `split_indices.csv` coincidieron con los valores de la sección de datos. El log sanitizado y la salida de verificación están en `evidencias/recuperacion/dagshub-remote-check.txt`.

Para que el docente recupere los datos sin usar las credenciales de la autora, debe configurar el remote HTTPS en un clon y autenticarse con su propio token de DagsHub (no se incluye ningún token en Git):

```bash
dvc remote add --local dagshub-public https://dagshub.com/alomariaDev/defiende-el-servidor.dvc
dvc remote modify --local dagshub-public auth basic
dvc remote modify --local dagshub-public user alomariaDev
dvc remote modify --local dagshub-public ask_password true
dvc pull -r dagshub-public data/raw.dvc data/processed.dvc
```

El comando solicita el token como contraseña de forma interactiva. La lectura anónima permanece pendiente de resolución/verificación; si el taller exige que no se solicite token al docente, este resultado debe considerarse una limitación del acceso actual.

## Backup DVC portable para recuperación

Se creó el remote DVC local `backup` en una carpeta hermana al repositorio. `dvc push -r backup data/raw.dvc data/processed.dvc` terminó con `7 files pushed`; `dvc status -c -r backup` respondió `Cache and remote 'backup' are in sync.` El archivo `dvc-backup.zip` contiene esos 7 objetos, pasó la prueba de integridad ZIP y ocupa 21.154.524 bytes. Su SHA-256 medido con `sha256sum` es `10a4833e607a86ad3d6a530fc70b0e5a3fd9593c6d9e2ba81810559746ef6b9`.

El respaldo se comparte mediante [este enlace de OneDrive](https://uexternadoedu-my.sharepoint.com/:u:/r/personal/maria_guzman17_est_uexternado_edu_co/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmaria%5Fguzman17%5Fest%5Fuexternado%5Fedu%5Fco%2FDocuments%2Fdvc%2Dbackup%2Ezip&parent=%2Fpersonal%2Fmaria%5Fguzman17%5Fest%5Fuexternado%5Fedu%5Fco%2FDocuments&share=cQrgaC5wUcuRRbzof4SjAgpiEgUC2Kp79Qm4FGZZ%5FPrzUSHEJg). La autora informa que descargó el archivo en incógnito. La comprobación automatizada directa de SharePoint devolvió redirecciones repetidas, pero luego se probó el archivo descargado por la autora: su SHA-256 coincide con el ZIP generado y, desde un clon temporal con el cache vacío y sin credenciales, `dvc pull -r backup` recuperó los datos; los cinco SHA-256 coincidieron. La salida está en `evidencias/recuperacion/onedrive-recovery-actual.log`; las limitaciones de la prueba automatizada original se conservan en `onedrive-check.txt`.

DagsHub requiere autenticación con token para recuperar los datos; el ZIP de OneDrive es la alternativa sin esas credenciales. El directorio `dvc-backup/` y el archivo `dvc-backup.zip` están fuera del repositorio y deben compartirse aparte del ZIP del proyecto. Descarga el respaldo de OneDrive, guárdalo en la carpeta padre del clon y, desde la raíz del repo, ejecuta:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m zipfile -e ../dvc-backup.zip ..
dvc remote add --local backup ../dvc-backup
dvc pull -r backup data/raw.dvc data/processed.dvc
sha256sum data/raw/UNSW_NB15_training-set.csv data/raw/UNSW_NB15_testing-set.csv
```

Los SHA-256 esperados son `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` (training) y `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` (testing). `--local` guarda la ruta únicamente en `.dvc/config.local`, fuera de la configuración versionada.

## Evaluación final única en test

Se evaluaron una vez las 82.332 filas del testing-set con v2 congelada y regla `y_pred=1` si `score_attack >= 0.3903680448122325`; no se descartaron filas ni se modificó el modelo/umbral. El test contiene label 0: 37.000 y label 1: 45.332.

- Hash SHA-256 del test: `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559`.
- Matriz [[TN,FP],[FN,TP]]: [[23531,13469],[1784,43548]].
- Recall ataque: 0.9606459013500397; precision ataque: 0.763772208288756; FPR: 0.36402702702702705; macro-F1: 0.8030990858090168; accuracy: 0.8147378904921537.
- Sobre validación, recall baja 0.018278974000863202, precision 0.16710153170088504 y macro-F1 0.1211855108054326; FPR sube 0.20911631274131276. Se reporta sin reajuste.
- Solapamiento por huella de las 42 entradas: 6.930 filas de test (1.047 grupos) coinciden con train; 1.611 filas (255 grupos) con validación; 0 filas/grupos coinciden con ambas particiones; train-val no comparte grupos.
- CSV por fila `evidencias/predicciones-test.csv`, SHA-256 `f02e69daa4ac9177f59cc26b378c25166c30df4a771218297ae770f6e4dfcb51`. Métricas `evidencias/metricas-test.json`, SHA-256 `7a3c1259357d327e3351fe668ff58c1539ad19ff944b628ace41ba30b3e014c3`.
- Código/commit de evaluación: `src/evaluate_test.py`, commit `89b481b42a35c6b9f21f2d0068f321bc6e02321e`; commit posterior que versionó esas evidencias: `3b816e3863ddc6c8e7dcb706db8d05e8d5d9f503`.

## Índice de evidencias

| Requisito | Evidencia |
|---|---|
| Auditoría de datos y contradicciones | `evidencias/auditoria.json` (SHA-256 `0afbe6bbe82a3f89b3765462cb887657142287b267437df5ad22a948c397c22e`) |
| Split, índices y agrupamiento | `evidencias/split.json`; `data/processed/split_indices.csv` |
| Experimentos y runs | `evidencias/experimentos.csv`; `evidencias/mlflow/` |
| Selección v2 y versión DVC | `evidencias/seleccion.json`; `data/processed.dvc` |
| Publicación y SHA del Hub | `evidencias/huggingface.json`; model cards en `evidencias/huggingface/{v1,v2}/` |
| Recuperación limpia | `evidencias/recuperacion/manifest.json` y `evidencias/recuperacion/run-20261025T195751Z/` |
| Predicción y métricas finales | `evidencias/predicciones-test.csv`; `evidencias/metricas-test.json` |
| Informe breve | `informe.md` |

## Declaración de uso de IA

Se utilizó Codex como apoyo para implementar y revisar scripts, resolver errores del flujo y redactar documentación. Los resultados numéricos, hashes, commits y run IDs de esta entrega se copiaron de los artefactos generados en el proyecto; no se inventaron ni se sustituyeron. La autora conserva la responsabilidad de verificar y presentar el trabajo.
