# Informe — Defiende el servidor

**Autora:** Maria Isabell Guzman  
**Problema:** clasificación binaria de tráfico UNSW-NB15 (normal=0, ataque=1)

## 1. Datos y preparación

El objetivo fue detectar ataques usando `label` como variable objetivo y 1 como clase positiva. El training-set tiene 175.341 filas y 45 columnas; se utilizaron 42 entradas predictoras, excluyendo `label`, `attack_cat` e `id`. La auditoría encontró cero faltantes, 56.000 ejemplos normales y 119.341 ataques. El SHA-256 del training es `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`.

Se identificaron 9.748 grupos de entradas idénticas, equivalentes a 74.301 ocurrencias adicionales sobre una primera fila, con grupo máximo de 415. En 229 grupos la misma entrada aparece con etiquetas distintas. La política fue preservar todas las observaciones y etiquetas, mantener cada grupo de predictores en una sola partición y dejar visibles las contradicciones, sin resolverlas mediante eliminación o relabeling. Con `StratifiedGroupKFold`, semilla 42, el primer fold separó train (140.274 filas: 44.800 normales y 95.474 ataques) de validación (35.067: 11.200 normales y 23.867 ataques), sin grupos compartidos. El test no intervino en la elección del modelo o el umbral.

El Pipeline ajusta imputador mediano y escalado estándar para numéricas, e imputador por moda y OneHotEncoder para categóricas, junto al estimador. Todo ajuste se hizo exclusivamente con train. Los datos originales y preparados se versionaron con DVC remoto accesible en DagsHub. Huellas de directorio: `fe93051ae209ffb0128a31e7a478c009.dir` (raw) y `a065edaae55501c53239ceade3a3a8e0.dir` (procesados).

## 2. Experimentos

Las cuatro configuraciones usaron el mismo split y semilla 42. Se seleccionó el umbral maximizando macro-F1 en scores de validación de cada ejecución. La matriz de abajo sigue el orden `[[TN, FP], [FN, TP]]`. Precision/recall/F1 usan `zero_division=0`; FPR es FP/(FP+TN), definido como 0 cuando no hay negativos reales.

| Configuración | Hipótesis/cambio frente a anterior | Umbral | Recall | Precision | FPR | Macro-F1 | Matriz |
|---|---|---:|---:|---:|---:|---:|---|
| LR C=1 (`lr_c1`) | Referencia lineal balanceada | 0.3903680448122325 | 0.9789248753509029 | 0.9308737399896411 | 0.1549107142857143 | 0.9242845966144494 | [[9465,1735],[503,23364]] |
| LR C=0.1 (`lr_c0_1`) | Más regularización que C=1 | 0.40173156466464144 | 0.9798047513302887 | 0.9286395044079104 | 0.16044642857142857 | 0.9227249711857688 | [[9403,1797],[482,23385]] |
| Árbol profundidad 5 (`dt_depth5`) | Nueva familia no lineal, profundidad limitada | 0.5302063907328993 | 0.9401265345456069 | 0.9565180322278114 | 0.09107142857142857 | 0.9204407774366041 | [[10180,1020],[1429,22438]] |
| Árbol profundidad 15 (`dt_depth15`) | Más capacidad que el árbol de profundidad 5 | 0.3537730527816762 | 0.9558805044622282 | 0.9655084853358161 | 0.07276785714285715 | 0.9390771244547116 | [[10385,815],[1053,22814]] |

El árbol de profundidad 15 consiguió el mayor macro-F1 de estas cuatro filas, además del menor FPR, mientras que LR C=1 obtuvo más recall. Esta comparación es relevante para interpretar la selección congelada: `seleccion.json` eligió LR C=1 como referencia para v1/v2 y optimizó su umbral en validación, no seleccionó el mejor macro-F1 global entre las cuatro configuraciones. La decisión v2 se preserva para respetar la congelación y se señala como limitación del proceso de selección.

## 3. Versiones y decisión del umbral

v1 quedó en el tag Git `v1`, commit `c2891e168109f7ab89dec90ce8d188ac687c1539`, con LR C=1 y umbral 0.5. En validación obtuvo recall 0.9539531570788118, precision 0.9479162329822224, FPR 0.11169642857142857 y macro-F1 0.9226488208398218; matriz [[9949,1251],[1099,22768]].

v2 quedó congelada en tag `v2`, commit `f83efdcbb4eb75db01b4f27a79ed4a6393cc6e3c`. El único cambio fue el umbral a 0.3903680448122325, elegido solo con la validación de LR C=1 para maximizar macro-F1; modelo y preprocesamiento no cambiaron. En validación, recall subió a 0.9789248753509029 y macro-F1 a 0.9242845966144494. Precision bajó a 0.9308737399896411 y FPR subió a 0.1549107142857143. Se aceptaron 484 falsos positivos adicionales para reducir 596 falsos negativos. La política favorece detectar más ataques, con costo de alertas normales adicionales.

El Pipeline completo se registró en MLflow local, run `ffdf524d57f749d28b5ed7fceb0fa0bf`, y se publicó en Hugging Face. SHA de revisión Hub v1: `04f723fe871e9b2f3b10b2ccd88cb2b45f587b53`; v2: `03e583e00e2c3d5dda0d0eb1fb9364ab1b86c2d6`. SHA-256 del artefacto: `e0f3019485ecfaa7418c7d4c31b7d130b1c5f003162de5f925d83561b7683dc3`.

## 4. Recuperación

En un checkout nuevo del commit `ec8b14923f3a5539250030045b47793c8637d4f8` se instalaron dependencias fijadas, se hizo pull de DVC y se verificaron hashes de datos. Se descargaron v1 y v2 por sus SHA completos del Hub y se cargaron los pipelines. Para una muestra fija de 25 filas de validación, coincidieron los row_id, etiquetas y clases predichas con las salidas previas. La diferencia absoluta máxima de probabilidades fue `2.220446049250313e-16` para cada versión, dentro de tolerancia `1e-12`. La bitácora, hashes y comparaciones están en `evidencias/recuperacion/run-20261025T195751Z/`.

## 5. Evaluación única en test

Después de congelar v2 se evaluó una vez el testing-set completo: 82.332 filas, sin descartar observaciones. Su SHA-256 es `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559`; clases: 37.000 normales y 45.332 ataques. La decisión fue ataque si score ≥ 0.3903680448122325. La matriz fue [[23531,13469],[1784,43548]]: TN=23.531, FP=13.469, FN=1.784, TP=43.548. Métricas: recall de ataque 0.9606459013500397, precision 0.763772208288756, FPR 0.36402702702702705, macro-F1 0.8030990858090168 y accuracy 0.8147378904921537.

Frente a validación con el mismo umbral, en test el recall bajó 0.018278974000863202, la precision bajó 0.16710153170088504, macro-F1 bajó 0.1211855108054326 y FPR aumentó 0.20911631274131276. No se reajustó el umbral ni el modelo. El solapamiento por huella de predictores fue de 6.930 filas de test (1.047 grupos) con train y 1.611 filas (255 grupos) con validación; no hubo filas/grupos compartidos con ambas. Las evidencias fila a fila y métricas están en `evidencias/predicciones-test.csv` y `evidencias/metricas-test.json`.

## 6. Limitaciones y conclusión

La caída de precision, macro-F1 y FPR observado en test muestra que el desempeño de validación no se trasladó completamente. El solapamiento de patrones idénticos entre test y los datos de desarrollo y las etiquetas contradictorias dentro de grupos son aspectos de los datos que dificultan interpretar generalización y techo de rendimiento. Además, solo se usó un split agrupado con semilla fija; no se reporta variación entre semillas ni calibración independiente. La selección de v2 se centró en la curva de umbral de LR C=1 y no fue la mejor puntuación global de las cuatro configuraciones. El umbral elegido eleva falsos positivos, por lo que su uso operacional requeriría valorar el costo de alertas y validar en el contexto de despliegue. Estas limitaciones no justifican alterar retrospectivamente el resultado final.

**Uso de IA:** Codex ayudó a escribir y depurar scripts, revisar coherencia de evidencias y redactar este borrador. Cifras, hashes, commits y run IDs se tomaron de los artefactos y salidas reales del workspace; no se fabricaron resultados. La autora es responsable de la revisión y entrega.
