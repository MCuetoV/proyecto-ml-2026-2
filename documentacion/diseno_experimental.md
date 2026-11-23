# Documento de diseño experimental

**Proyecto:** Machine Learning (650064), ULima 2026-2 · **Entrega:** viernes 27/11/2026
**Versión:** 1.5 (aprobada; 1.2 corrige la regla 9 de la sección 12; 1.3 ajusta la partición agrupada en §7 y las funciones de §11.3; 1.4 fija el criterio del test de obesidad y agrega `codificar_objetivo`; 1.5 agrega el barrido de `C` del selector L1 como análisis de sensibilidad (§6) y quita las menciones residuales a los extras descartados) · **Fecha:** 22/11/2026
**Base:** `enunciado_proyecto_ML.md`, `2026-11-17_analisis-consigna-proyecto-ML.md` (secciones 2, 9, 10 y 11) y verificación directa de los CSV oficiales de UCI.
**Estado:** aprobado por Marcelo el 22/11/2026 e implementado (Fase 3 cerrada). Decisiones: Git todo en `main`; reparto de la sección 13; **extras descartados** (§14); barrido de `C` del selector L1 aprobado el 22/11 (v1.5).

---

## Resumen (una página)

| # | Decisión | Elección |
|---|---|---|
| 1 | Escenarios | Obesidad: **E1 con** Height/Weight y **E2 sin** ellos (principal: **E2**). Estudiantes: **E1 matrícula** (24 variables) y **E2 fin del 2.º semestre** (36) (principal: **E1**). |
| 2 | Métodos nuevos | **MLP + KNN** en ambos datasets. |
| 3 | Referencias | `DummyClassifier` + método de los autores reproducido: **árbol de decisión (CART ≈ J48)** en obesidad y **Random Forest con sobremuestreo** en estudiantes, ambos ajustados con `GridSearchCV`, como hicieron los autores. |
| 4 | MLP | `MLPClassifier` de scikit-learn con *early stopping*. |
| 5 | Preprocesamiento | Estudiantes: *one-hot* con agrupación de niveles raros. Obesidad: eliminar 24 duplicados exactos y **redondear a la precisión de la encuesta** (borra la huella de SMOTE). Escalado estándar. Todo dentro del `Pipeline`. |
| 6 | Desbalance | Estudiantes: sin balanceo vs. **SMOTENC** dentro de cada *fold*. Obesidad: no se remuestrea. |
| 7 | Selección de variables | **L1 (Lasso logístico)** con `SelectFromModel` dentro del `Pipeline`, comparado contra todas las variables, más un **barrido de `C`** como análisis de sensibilidad (v1.5). |
| 8 | Validación | Test de reserva 80/20 estratificado, fijo y usado una vez. Optuna con CV de 5 *folds*; comparativa con CV 5×3. En obesidad, **partición agrupada** para que filas idénticas no queden a ambos lados. Sin *nested CV*. |
| 9 | Optuna | TPE, semilla 42; 50 *trials* en KNN y 100 en MLP (cabe en Colab: ~4 s por evaluación). `GridSearchCV` vs. Optuna en KNN. |
| 10 | Métricas | Principal: **macro-F1** en ambos. Obesidad: **kappa ponderado cuadrático** como segunda métrica obligatoria. |
| 11 | Interpretación | Importancia por permutación, matriz de confusión, curva de aprendizaje y Wilcoxon (con su limitación declarada). |
| 12 | Git | **Todo en `main`**, *commits* pequeños, un responsable por archivo; resultados versionados por la misma corrida que genera el notebook. |
| 13 | Calendario | Fase 2 el lunes 23; código el 24 y 25; reporte completo el jueves 26; control final el viernes 27. |

**Hallazgos de los datos que cambian el diseño (sección 0):**
1. En obesidad, la clase **Normal_Weight es 100% real**, y Obesity_Type_III es 99% sintética (3 filas reales de 324). Los decimales delatan la clase; por eso se redondea.
2. Tras redondear y sin Height/Weight, **el 96% de las filas de Obesity_Type_III son copias exactas** de otras. Por eso la partición es agrupada.
3. Una prueba preliminar (modelos sin ajustar, solo para dimensionar) mostró que los decimales y las copias inflan el macro-F1 de KNN en E2 en **unos 12 puntos** (0.723 frente a 0.600). Eso se convierte en un resultado del reporte.

**Aprobado el 22/11:** Git en `main` (12), reparto de la sección 13 y resto del diseño. Extras descartados por decisión del equipo (§14).

---

## 0. Hechos verificados con los CSV oficiales (22/11)

Archivos en `datos/` sin modificar: `data.csv` (MD5 `ee29c9ee…`) y `ObesityDataSet_raw_and_data_sinthetic.csv` (MD5 `aff636fc…`).

### Estudiantes
- 4424 × 37, sin nulos y sin duplicados. Separador `;`, codificación con BOM (`utf-8-sig`) y una columna con tabulación al final del nombre (`"Daytime/evening attendance\t"`): el cargador limpia los nombres.
- Target: Graduate 2209, Dropout 1421, Enrolled 794.
- Las 3 variables macroeconómicas forman **solo 10 combinaciones**: confirman que funcionan como cohorte (años 2008/09 a 2018/19).
- Variables nominales con muchos niveles raros (< 1% de filas): Nacionality 20 de 21, Father's occupation 34 de 46, Father's qualification 27 de 34, Mother's qualification 21 de 29, Mother's occupation 22 de 32, Previous qualification 12 de 17, Application mode 9 de 18.
- 180 estudiantes no inscribieron unidades en el 1.º semestre (77 Dropout, 75 Graduate, 28 Enrolled): los ceros son legítimos.
- Realinho et al. (2022) clasifican Debtor y Tuition fees up to date como socioeconómicas, pero **no dicen cuándo se midieron**. Si se registraron después de la matrícula, serían fuga en E1. Se declara como limitación en la Discusión.

### Obesidad
- 2111 × 17, sin nulos, **24 filas duplicadas** (33 filas involucradas, todas con apariencia real).
- Clases entre 272 y 351 filas.
- **Huella de SMOTE:** 491 filas tienen enteros en FCVC, NCP, CH2O, FAF, TUE y Age (el descriptor declara 485 reales). La proporción real por clase es: Normal_Weight 100%, Overweight I 24%, Overweight II 20%, Insufficient 15%, Obesity I 13%, Obesity II 4%, Obesity III 1%. En las filas reales, Height tiene 2 decimales (99.6%) y Weight es entero (96.5%); en las sintéticas casi nunca.
- **Artefacto de género confirmado:** Obesity_Type_II tiene 295 hombres de 297 y Obesity_Type_III, 323 mujeres de 324.
- **Menores de edad:** 113 filas con Age < 18 (unas 38 reales).
- **La etiqueta no es una función limpia del IMC:** los rangos de IMC por clase se solapan (por ejemplo, Overweight_Level_I 22.8–28.8 y Normal_Weight 18.5–24.9). Por eso ni siquiera E1 llega al 100%.
- Niveles raros: CALC = "Always" tiene **1 fila**; MTRANS Bike 7 y Motorbike 11.
- Tras redondear y sin Height/Weight, 895 filas pertenecen a grupos de vectores idénticos (667 repeticiones). Obesity_Type_III aporta 312 de ellas: solo tiene 50 vectores únicos en 324 filas. Hay 28 vectores que aparecen con etiquetas distintas.

### Prueba de costo y de protocolo (preliminar, no son resultados del reporte)
Modelos sin ajustar, 5 *folds*, 2 CPU como Colab gratuito.
- Estudiantes: una evaluación de MLP con 5 *folds* tarda 1.5–4 s; KNN, 0.2–1.2 s. Tras el *one-hot* quedan 106 columnas.
- Obesidad E2, macro-F1 de KNN: 0.723 sin redondear con partición estratificada, 0.620 redondeado con partición estratificada y 0.600 redondeado con partición agrupada. MLP: 0.700, 0.673 y 0.659.

---

## 1. Escenarios por dataset

| Dataset | Escenario | Variables | Rol |
|---|---|---|---|
| Obesidad | **E1: con antropometría** | Las 16 | Referencia: reproduce la condición de los autores y muestra que el problema se reduce a reconstruir el IMC |
| Obesidad | **E2: sin antropometría** | 14: Gender, Age, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS | **Principal**: responde la pregunta del título del dataset (hábitos y condición física) |
| Estudiantes | **E1: al matricularse** | 24: las 36 menos las 12 *Curricular units* (1.º y 2.º semestre) | **Principal**: caso de uso de alerta temprana y única condición comparable con Martins et al. (2021), que solo usaron información de matrícula |
| Estudiantes | **E2: fin del 2.º semestre** | Las 36 | Cota superior: cuánto aporta el desempeño académico inicial |

- **Opciones descartadas:** un solo escenario por dataset. En obesidad, el número con Height/Weight es engañoso por sí solo; en estudiantes, el escenario completo mezcla información que no existe al matricularse.
- **Escenario principal = donde se hacen las ablaciones** (balanceo y selección) y la interpretación completa. En ambos escenarios se ajustan y evalúan todos los modelos.
- **Riesgos:** los números del escenario principal son más bajos y hay que explicarlos bien. En estudiantes E1, Debtor y Tuition fees podrían ser posteriores a la matrícula.

## 2. Métodos

### 2.1 Métodos nuevos

| Método | Estudiantes | Obesidad | Justificación ligada a los datos | Base |
|---|---|---|---|---|
| **KNN** | ✅ limpio | ✅ limpio | Modelo basado en distancias, que contrasta con los árboles, los lineales y el *boosting* de los autores. Es sensible a la codificación y al escalado, que son temas del curso y trampas de estos datos. Es el único método del curso que nadie usó en ningún caso | Curso (imputador y modelo de prueba); como clasificador principal es una extensión menor |
| **MLP** | ✅ limpio | ✅ defendible (declarar Molina et al., 2021) | Datos tabulares → MLP (PPT de tipos de redes). Capta interacciones no lineales entre hábitos sin especificarlas. Su ajuste con Optuna cumple la estrategia (b) | Curso (redes neuronales) |

- **Descartados:** SVM y RF en obesidad (Molina et al. los usaron); AdaBoost en estudiantes (frontera por LogitBoost) y en obesidad mientras no se dicte; LDA/QDA y regresión ordinal (no vistos en el curso).
- **Elección:** MLP + KNN en ambos datasets. XGBoost, que figuraba como tercer método opcional en obesidad, se descartó el 22/11 junto con los demás extras (§14).

### 2.2 Modelos de referencia (mismas particiones, mismo preprocesamiento, mismas métricas)
- `DummyClassifier(strategy="most_frequent")`: el piso.
- **Obesidad:** `DecisionTreeClassifier(criterion="entropy")` como aproximación a J48. Se declara que CART no es C4.5: CART hace cortes binarios y otra poda, y C4.5 usa *gain ratio* y cortes múltiples en categóricas. Se ajusta con `GridSearchCV` (`max_depth` ∈ {3, 5, 8, 12, None}; `min_samples_leaf` ∈ {1, 5, 10}).
- **Estudiantes:** `RandomForestClassifier` con el mismo sobremuestreo que los modelos nuevos y `GridSearchCV` (`n_estimators` ∈ {200, 500}; `max_depth` ∈ {None, 10, 20}; `max_features` ∈ {"sqrt", 0.5}). Es el método de referencia que los mismos autores eligieron en su trabajo de 2023 y está en el curso.

## 3. Implementación del MLP

| | PyTorch | `MLPClassifier` |
|---|---|---|
| Integración con `Pipeline`, CV y Optuna | Manual | Directa |
| Explicabilidad para el equipo | Bucle de entrenamiento propio | Una línea por modelo |
| Lo que se pierde | — | *Dropout* y AdamW. Se compensa con la regularización L2 (`alpha`, equivalente al *weight decay*) y con *early stopping* |

**Elección:** `MLPClassifier(early_stopping=True, validation_fraction=0.1, max_iter=500, random_state=42)`. En Antecedentes se conecta con la teoría vista en clase: capas, ReLU/tanh, *backpropagation*, Adam, SGD con *momentum*, mini-lotes y L2.
**Riesgo:** avisos de no convergencia. Se registran y, si aparecen en la mejor configuración, se sube `max_iter`.

## 4. Preprocesamiento

### 4.1 Pasos determinísticos fila a fila (antes de dividir; no aprenden de los datos, así que no hay fuga)
- **Obesidad:**
  1. Eliminar los 24 duplicados exactos del CSV original, para que una misma respuesta no quede en train y en test.
  2. Calcular la marca `probablemente_real` con los valores originales: enteros en FCVC, NCP, CH2O, FAF, TUE y Age. **No es una variable del modelo**; solo sirve para el EDA y para la evaluación de sensibilidad (sección 7).
  3. Redondear FCVC, NCP, CH2O, FAF, TUE y Age al entero, Height a 2 decimales y Weight al entero. **Por qué:** son respuestas discretas de una encuesta; los decimales los creó SMOTE y, como la proporción sintética varía de 0% a 99% según la clase, delatan la etiqueta.
  4. Construir `grupo`: el identificador de las filas idénticas en las 14 variables sin antropometría ya redondeadas. Se usa para particionar (sección 7).
- **Estudiantes:** limpiar los nombres de columnas. No hay duplicados.

### 4.2 Pasos dentro del `Pipeline` (aprenden solo del *fold* de entrenamiento)

| Tipo de variable | Estudiantes | Obesidad | Transformación |
|---|---|---|---|
| Nominal | Marital status, Application mode, Course, Previous qualification, Nacionality, ocupación y calificación de la madre y del padre (9) | MTRANS | `OneHotEncoder(min_frequency=0.01, handle_unknown="infrequent_if_exist")`: los niveles con menos del 1% se agrupan en "infrecuente" |
| Binaria | Daytime/evening, Displaced, Educational special needs, Debtor, Tuition fees, Gender, Scholarship holder, International | Gender, family_history, FAVC, SMOKE, SCC | Ya están en 0/1 (estudiantes) o `OneHotEncoder(drop="if_binary")` (obesidad) |
| Ordinal | Application order (0–9) | CAEC y CALC (no < Sometimes < Frequently < Always); FCVC, NCP, CH2O, FAF y TUE ya redondeadas | `OrdinalEncoder` con orden explícito y luego escalado |
| Numérica | Notas, edad, *Curricular units* (E2) y macroeconómicas | Age, Height y Weight (E1) | `StandardScaler` |

- **Descartados:** `BinaryEncoder` (crea parecidos falsos entre niveles y distorsiona las distancias de KNN) y *target encoding* (no visto en el curso; riesgo de fuga).
- **Atípicos:** IQR e Isolation Forest **solo como diagnóstico** en los notebooks de exploración. No se eliminan filas: en estudiantes los extremos son plausibles (ceros legítimos) y en obesidad son artefactos sintéticos que ya se tratan con el redondeo y la partición agrupada.
- **Riesgo:** CALC = "Always" tiene 1 fila; con *min_frequency* no aplica porque CALC es ordinal. Se fusiona "Always" con "Frequently" en el paso determinístico y se declara.

## 5. Desbalance (solo estudiantes)

- **Base del curso:** SMOTE aplicado solo al train.
- **Limitación observada:** 9 variables nominales en *one-hot*. SMOTE interpola y produce columnas *dummy* con valores como 0.37, que no corresponden a ninguna categoría.
- **Extensión justificada:** **SMOTENC** (Chawla et al., 2002, sección de datos mixtos). Interpola las numéricas igual que SMOTE y, en las nominales, asigna la categoría más frecuente entre los vecinos.
- **Ubicación:** primer paso del `Pipeline` de imblearn, así que se aplica dentro de cada *fold* y nunca toca la validación ni el test.
- **Comparación** en el escenario principal, con CV 5×3: sin balanceo vs. SMOTENC, para KNN y MLP. Además, el balanceo entra como hiperparámetro en Optuna (sí/no). **`class_weight` no aplica** a KNN ni a MLP en scikit-learn; solo se prueba en el RF de referencia.
- **Obesidad:** no se remuestrea, porque ya fue balanceado con SMOTE por los autores.
- **Riesgo:** con *early stopping*, la validación interna del MLP incluye filas sintéticas. El efecto es menor; se declara.

## 6. Selección de variables

- **Método principal:** `SelectFromModel(LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000))`, es decir, Lasso para clasificación, visto en clase. Va después del preprocesamiento y antes del modelo, **dentro del `Pipeline`**, así que se aprende en cada *fold*.
- **Comparación:** la mejor configuración de KNN y de MLP, con y sin selector, en el escenario principal, con CV 5×3. Se reporta el número de variables retenidas.
- **Diagnóstico complementario:** la importancia por permutación (sección 10) muestra qué variables originales pesan, sin decidir nada sobre ellas.
- **Descartados como paso del `Pipeline`:** Boruta, porque reentrena muchos RF en cada *fold* (costoso), y chi², que exige valores no negativos y choca con el escalado.
- **Riesgo:** en obesidad E2 hay pocas variables y la selección puede no ayudar. Ese también es un resultado válido.
- **Barrido de `C` (v1.5, decisión del 22/11).** Con `C = 1` el selector casi no elimina columnas (ajustado con todo el train: 90 de 94 en estudiantes E1 y 17 de 17 en obesidad E2), así que la selección quedaba cumplida solo de nombre. Se agrega un análisis de sensibilidad en la sección 7.1 de los notebooks 02 y 04:
  - `C` ∈ {0.01, 0.05, 0.1, 0.5, 1}, con `C` = 1/λ (visto en clase con la regresión logística): cuanto menor es `C`, más coeficientes lleva Lasso a cero. El extremo inferior se fijó con una prueba previa que solo contó columnas: con `C` < 0.01 algunos *folds* se quedan sin columnas.
  - Misma configuración de KNN y MLP que la comparación anterior, escenario principal y los mismos 15 *folds* de la comparativa (CV 5×3, semilla 43).
  - Se reportan el macro-F1 (media ± desviación) y las columnas conservadas por *fold* (media, mínimo y máximo), contadas dentro de cada *fold* con `cross_validate(..., return_estimator=True)`, nunca ajustando el selector fuera de la CV. Con `C = 1` el resultado debe repetir la comparación anterior; el notebook imprime la diferencia como control.
  - **Criterio fijado antes de ver resultados:** el rendimiento se mantiene si el macro-F1 medio queda a menos de una desviación entre *folds* del valor con `C = 1`. Si se cumple con menos columnas, se reporta como evidencia de que se puede seleccionar sin pérdida; si no, queda justificado conservar las variables.
  - **No cambia los modelos finales, la comparativa ni el test.** `C = 1` se fijó en el diseño antes de ver resultados; cambiarlo ahora obligaría a recalcular todo y podría leerse como un ajuste a posteriori.
  - Nota técnica: `saga` es estocástico y el selector no fija `random_state`; en una prueba repetida, el macro-F1 varió solo en el cuarto decimal.
  - Salidas: `tabla_08_seleccion_c_{…}.csv` y `figura_08_seleccion_c_{…}.png` (§11.2), más filas en el registro largo.

## 7. Protocolo de validación

1. **Test de reserva 20%, estratificado y con semilla 42**, creado una sola vez por una función de `utilidades.py`, para que los notebooks de exploración y de modelado usen la misma partición.
   - Obesidad: partición **estratificada y agrupada** (`StratifiedGroupKFold(5)`; el primer *fold* es el test). Las filas idénticas tras redondear quedan del mismo lado. **El mismo test sirve a E1 y E2**, así los escenarios son comparables.
   - *Ajuste de la Fase 2:* con `shuffle=True` la estratificación falla, porque Obesity_Type_III tiene grupos de hasta 75 filas idénticas: el test quedaba con 22.7% de esa clase frente a 15.5% en el total. Sin barajar, el algoritmo reparte mucho mejor. Para obtener particiones distintas con cada semilla, los grupos se renumeran al azar antes de dividir (`utilidades.particiones_agrupadas`). Resultado verificado: test de 426 filas (20.4%), desviación máxima por clase de 2.1 puntos, ningún grupo compartido entre train y test; en la CV, desviaciones de 0.3 puntos como máximo y particiones distintas en cada repetición.
   - *Ajuste de la Fase 3:* con el primer *fold* como test, toda la clase Obesity_Type_III del test era **un único vector repetido 75 veces** (el grupo más grande del dataset), lo que volvía inútil su evaluación. Regla nueva, estructural y fijada antes de entrenar cualquier modelo: de las 5 particiones agrupadas, el test es la de distribución de clases más parecida a la del total. Resultado: test de 416 filas (19.9%), desviación máxima por clase de 0.5 puntos, 13 vectores distintos de Obesity_Type_III y ningún grupo compartido.
   - Estudiantes: `train_test_split(stratify=y)`.
2. **El EDA que informa decisiones se hace solo con el train.** La tabla descriptiva global del dataset puede usar todas las filas.
3. **Optuna:** CV de 5 *folds* sobre el train (`StratifiedKFold` o `particiones_agrupadas` en obesidad), con semilla 42.
4. **Comparativa:** CV de 5 *folds* repetida 3 veces, con **otra semilla (43)**, para todas las configuraciones finales: 15 valores por modelo.
5. **Test:** cada configuración final se reentrena con todo el train y se evalúa **una sola vez**. Se reportan todos los modelos, no solo el mejor, y no se cambia nada después de verlo.
6. **Sensibilidad del protocolo (obesidad, E2):**
   - (a) protocolo ingenuo: sin redondear y con partición estratificada, contra el protocolo elegido;
   - (b) métricas del test restringidas a las filas `probablemente_real`, con su conteo por clase. Hay muy pocas reales en Obesity II y III, así que se presenta como descriptivo.
- ***Nested CV*: descartado.** El test de reserva ya da una estimación sin sesgo del modelo final. El sesgo de seleccionar con la misma CV se reduce usando otra semilla en la comparativa y se declara como limitación.

## 8. Ajuste de hiperparámetros con Optuna

- **Sampler:** `TPESampler(seed=42)`. **Objetivo:** maximizar el macro-F1 medio de la CV de 5 *folds* sobre el train.
- **Presupuesto** (medido: MLP ≈ 1.5–4 s por evaluación de 5 *folds* en 2 CPU): **KNN 50 trials** (< 1 min); **MLP 100 trials** (unos 5–10 min por escenario). Total por dataset: menos de 30 min. Con `timeout` como red de seguridad.

| Modelo | Espacio de búsqueda |
|---|---|
| KNN | `n_neighbors` 1–50; `weights` {uniform, distance}; `p` {1 Manhattan, 2 Euclídea}; balanceo {no, SMOTENC} en estudiantes |
| MLP | capas {1, 2, 3}; neuronas por capa 16–256 (escala log); `activation` {relu, tanh}; `solver` {adam, sgd con momentum 0.9}; `alpha` 1e-6–1e-1 (log); `learning_rate_init` 1e-4–1e-1 (log); `batch_size` {32, 64, 128, 256}; balanceo {no, SMOTENC} en estudiantes |

- **Comparación con `GridSearchCV` (visto en clase), en KNN y en el escenario principal:** malla completa (50 × 2 × 2 = 200 combinaciones) contra 50 *trials* de Optuna. Se comparan el mejor macro-F1, el número de evaluaciones y el tiempo. Para el MLP, una malla mínima del espacio tendría unas 3000 combinaciones (más de 3 h con el costo medido): ese número se reporta como argumento de por qué se necesita una búsqueda eficiente.
- **Cumplimiento:** estrategia (a) = Optuna en KNN; estrategia (b) = Optuna en el MLP.
- **Riesgo:** los *trials* del MLP con 3 capas de 256 neuronas y lote 32 son los más lentos. Si una evaluación supera 20 s, se acota el espacio y se registra.

## 9. Métricas

| | Estudiantes | Obesidad |
|---|---|---|
| **Principal** (objetivo de Optuna y criterio de elección) | **macro-F1**: trata por igual a Enrolled, la clase minoritaria y difícil; es comparable con el F1 promedio de Martins et al. | **macro-F1**, por coherencia entre datasets |
| Complementarias | F1 por clase, *balanced accuracy*, ROC-AUC OvR (macro) | **Kappa ponderado cuadrático** (obligatoria: penaliza más confundir clases lejanas del orden), F1 por clase |
| Del curso | Matriz de confusión, *accuracy*, *precision* y *recall* por clase | Ídem |

- **Alternativa considerada para obesidad:** kappa como métrica principal. Se descarta como objetivo porque el curso no la vio y porque mezclar objetivos complica la comparativa entre datasets; se reporta siempre junto al macro-F1 y se interpreta en la Discusión.
- **Orden de las clases de obesidad** para el kappa: Insufficient < Normal < Overweight I < Overweight II < Obesity I < Obesity II < Obesity III.

## 10. Comparativa e interpretación

- **Tabla comparativa por escenario:** media ± desviación (CV 5×3) de cada modelo, más el resultado en el test.
- **Prueba estadística:** Wilcoxon de rangos con signo entre el mejor modelo nuevo y la referencia de los autores, sobre los 15 *folds* pareados. Limitación declarada: los *folds* de una CV repetida no son independientes, así que el valor *p* es orientativo.
- **Interpretación del mejor modelo en el escenario principal:**
  - matriz de confusión en el test;
  - **importancia por permutación** sobre el test y por variable original, permutando la columna antes del `Pipeline` (Realinho et al. usaron la misma técnica, lo que permite contrastar);
  - **curva de aprendizaje** (`learning_curve`) para hablar de sesgo y varianza.
- **Puntos que la interpretación debe revisar explícitamente:** el peso de Gender en obesidad E2 (artefacto) y el de las variables macroeconómicas en estudiantes (cohorte).
- **Descartado:** SHAP, porque con KNN y MLP exige `KernelExplainer`, que es lento, y no agrega un tema del curso (§14).

## 11. Notebooks, tablas y figuras

### 11.1 Secciones (idénticas en los notebooks gemelos)

**01 y 03 (exploración):** 0. Configuración y autores · 1. Carga y descripción del dataset · 2. Valores faltantes y duplicados · 3. Partición train/test (función común) · 4. Análisis univariante y del target (train) · 5. Análisis multivariante y trampas del dataset (train) · 6. Atípicos (diagnóstico) · 7. Decisiones de preprocesamiento que se derivan.

**02 y 04 (modelado):** 0. Configuración y autores · 1. Carga, limpieza determinística, escenarios y partición · 2. `Pipeline` de preprocesamiento · 3. Modelos de referencia (Dummy y autores, con `GridSearchCV`) · 4. Optuna: KNN y MLP por escenario · 5. `GridSearchCV` vs. Optuna · 6. Balanceo (en obesidad: "no aplica", con su justificación) · 7. Selección de variables (7.1: barrido de `C`) · 8. Comparativa por CV (en 04, 8.1: sensibilidad del protocolo) · 9. Evaluación final en el test · 10. Interpretación · 11. Registro de resultados.

### 11.2 Archivos que se generan

El número es el de la tabla o figura candidata en el reporte; el sufijo indica el dataset. Por el límite de páginas, el reporte usará solo las marcadas con **R**.

| Archivo | Notebook | Contenido | R |
|---|---|---|---|
| `tabla_01_descripcion_{estudiantes,obesidad}.csv` | 01, 03 | Variables, tipos, escenarios y distribución del target | **R** |
| `tabla_02_hiperparametros_{…}.csv` | 02, 04 | Espacios de búsqueda y valores óptimos | **R** |
| `tabla_03_comparativa_cv_{…}.csv` | 02, 04 | Media ± desviación por modelo y escenario | **R** |
| `tabla_04_metricas_test_{…}.csv` | 02, 04 | Métricas en el test | **R** |
| `tabla_05_grid_vs_optuna_{…}.csv` | 02, 04 | Calidad, evaluaciones y tiempo | **R** |
| `tabla_06_ablaciones_{…}.csv` | 02, 04 | Balanceo y selección | **R** (fusionada con la 03 si falta espacio) |
| `tabla_07_sensibilidad_protocolo_obesidad.csv` | 04 | Protocolo ingenuo vs. elegido; test en filas reales | **R** |
| `tabla_08_seleccion_c_{…}.csv` | 02, 04 | Barrido de `C` del selector L1: macro-F1 y columnas conservadas por *fold* (v1.5) | **R** (fusionada con la 06 si falta espacio) |
| `figura_01_flujo_trabajo.png` | Script aparte (no son datos) | Figura de la Metodología | **R** |
| `figura_02_target_{…}.png` | 01, 03 | Distribución del target | **R** (combinada) |
| `figura_03_huella_sintetica_obesidad.png` | 03 | Proporción real por clase y sexo por clase | **R** |
| `figura_04_confusion_{…}.png` | 02, 04 | Matriz de confusión del mejor modelo en el test | **R** |
| `figura_05_importancia_{…}.png` | 02, 04 | Importancia por permutación | **R** |
| `figura_06_curva_aprendizaje_{…}.png` | 02, 04 | Curva de aprendizaje | Solo notebook |
| `figura_07_optuna_historial_{…}.png` | 02, 04 | Historial de Optuna del MLP | Solo notebook |
| `figura_08_seleccion_c_{…}.png` | 02, 04 | Macro-F1 y columnas conservadas según `C` (v1.5; el número quedó libre al descartar NSGA-II) | Si hay espacio |
| `resultados_{estudiantes,obesidad}.csv` | 02, 04 | Registro largo: dataset, escenario, modelo, configuración, métrica, media, desviación, partición | — |
| `resultados_experimentos.csv` | `utilidades.consolidar_resultados()` | Une los dos anteriores | — |

Cada notebook escribe **su propio** archivo de resultados para que dos personas nunca editen el mismo archivo.

### 11.3 Funciones comunes de `utilidades.py` (pocas y simples)
`SEMILLA = 42` · `cargar_estudiantes()` · `cargar_obesidad()` (incluye duplicados, marca real, redondeo y grupo) · `columnas_por_escenario(dataset, escenario)` · `codificar_objetivo(y, dataset)` (*Fase 3:* clases como enteros; en obesidad, en orden de severidad para el kappa; necesario porque `MLPClassifier` de scikit-learn 1.6.1 falla con *early stopping* y clases de texto) · `particiones_agrupadas(y, grupos, semilla)` · `dividir_train_test(datos, dataset)` · `crear_preprocesador(dataset, escenario)` · `evaluar_cv(modelo, X, y, grupos=None, repeticiones=1, semilla=SEMILLA)` · `metricas_test(modelo, X, y, dataset)` · `registrar_resultado(filas, ...)` · `guardar_tabla(tabla, nombre)` · `guardar_figura(figura, nombre)` · `consolidar_resultados()`. *Fase 2:* se agregaron `particiones_agrupadas` (ajuste de §7) y `guardar_tabla` (escritura uniforme de las tablas); `registrar_resultado` acumula filas en una lista y el notebook las guarda una sola vez al final, para que volver a ejecutarlo no duplique resultados. Las funciones objetivo de Optuna se quedan **en los notebooks**, para que los espacios de búsqueda se vean donde se explican.

## 12. Flujo de trabajo con Git (aprobado: todo en `main`)

**Opciones consideradas:** (A) todo en `main` con un responsable por archivo; (B) una rama por integrante con *pull request*. **Elegida: A**, por simplicidad para quienes trabajan en Colab. Como cada archivo tiene un solo responsable, no hay conflictos de fusión.

**Reglas:**
1. **Un archivo, un responsable** (sección 13). Nadie edita el archivo de otro; los cambios se piden.
2. **Antes de trabajar, siempre la última versión:** `git pull` en VS Code; en Colab, la celda 0 ya hace `git pull`, y el notebook se abre desde GitHub, nunca desde una pestaña vieja.
3. **Commits pequeños y descriptivos:** imperativo, en español y con el ámbito delante, por ejemplo: `02-estudiantes: agrega Optuna para KNN y MLP`. Un *commit* por bloque lógico, no uno por día.
4. **Resultados:** la corrida oficial que llena las salidas del notebook es la misma que genera sus archivos en `resultados/`, para que las cifras del reporte coincidan exactamente con las del notebook. En Colab, una celda final hace `git add` solo de los archivos de ese notebook, `commit`, `git pull --rebase` y `push` a `main`, con la identidad del integrante guardada en Secrets (probado con un remoto que avanzó entre medio). Después el notebook se guarda con "Save a copy in GitHub".
5. **Archivos de Marcelo:** `utilidades.py`, `requirements.txt`, `README.md`, `.gitignore`, `documentacion/` y `reporte/*.docx`. El .docx es binario y no admite fusión.
6. **`.gitignore`:** agregar `.DS_Store`, `~$*.docx` (archivos de bloqueo de Word) y `*.tmp`. Los CSV de `datos/` **sí se versionan**: pesan menos de 1 MB y tienen licencia CC BY 4.0, lo que hace el proyecto reproducible.
7. **Revisión cruzada:** Marcelo revisa cada notebook contra este diseño (por ejemplo, con `git log -p notebooks/02_…`) antes de dar por cerrado cada bloque, y anota las correcciones como *commits* del responsable.
8. **Hitos con etiquetas:** `v0-diseno`, `v1-codigo` y `v2-entrega`.
9. **Los *commits* no incluyen coautorías ni menciones a herramientas de IA.** El uso de IA se declara únicamente en la sección 9 del reporte.
- **Riesgo:** un `push` rechazado porque `main` avanzó. Se resuelve con `git pull --rebase` y repitiendo el `push`; como cada uno toca archivos distintos, no hay conflictos.

## 13. Reparto de responsables (aprobado)

| Integrante | Código | Reporte |
|---|---|---|
| **Marcelo** | `utilidades.py`, coordinación, revisión cruzada y `04_obesidad_modelado` | Metodología (figura del flujo), Introducción, Conclusiones, Declaración de uso de IA e integración final |
| **Eros** | `03_obesidad_exploracion` | Antecedentes (datasets y trabajos previos), Experimentación: descripción y EDA de obesidad, párrafo ético de obesidad |
| **Eduardo** | `02_estudiantes_modelado` | Antecedentes (KNN, MLP, Optuna y SMOTENC), Experimentación: modelado de estudiantes, Resultados de estudiantes |
| **Edson** | `01_estudiantes_exploracion` | Experimentación: descripción y EDA de estudiantes, párrafo ético de estudiantes, Resultados de obesidad |

- Discusión: la escriben juntos los dos responsables del modelado; Marcelo integra.
- Aportes: se redacta al final a partir del historial de *commits*.
- Claude escribe o acompaña el código y el texto según el reparto; cada responsable debe poder explicar cada línea de su notebook.

## 14. Cronograma y alcance

| Día | Código | Reporte |
|---|---|---|
| **Dom 22** | Aprobación del diseño | — |
| **Lun 23** (clase) | Fase 2 (adelantada al domingo 22): prueba local y en Colab | Antecedentes y Metodología (borrador) |
| **Mar 24** | 01 y 03 completos; 02 y 04 hasta las referencias | Experimentación: descripción de los datasets |
| **Mié 25** (clase) | 02 y 04: Optuna, ablaciones, comparativa; revisión cruzada | Experimentación completa |
| **Jue 26** | Test final, interpretación, corrida oficial y consolidación | Resultados, Discusión (con ética), Conclusiones, Introducción, Aportes, IA; revisión con los tres skills |
| **Vie 27** | Control de calidad final (Fase 5), etiqueta `v2-entrega`, acceso del profesor | Ajustes finales y envío |

**Alcance mínimo garantizado:** 2 escenarios × 2 datasets; Dummy + 1 método de los autores; KNN y MLP con Optuna; `GridSearchCV` vs. Optuna en KNN; balanceo en estudiantes; selección L1 con barrido de `C` (v1.5); CV 5×3; test final; matriz de confusión; importancia por permutación; curva de aprendizaje; Wilcoxon; sensibilidad del protocolo en obesidad.

**Extras: descartados por decisión del equipo (22/11).** El alcance mínimo quedó cumplido y el equipo decidió no agregar temas nuevos, para que el trabajo se mantenga dentro de lo visto en el curso y cualquier integrante pueda explicarlo. Lista original, solo como referencia:
1. XGBoost en obesidad.
2. NSGA-II en el MLP.
3. XGBoost como segunda referencia en estudiantes.
4. Sensibilidad sin Gender (obesidad) y sin Debtor ni Tuition fees (estudiantes E1).
5. RF como segundo selector de variables.
6. Friedman.
7. *Nested CV* en estudiantes E1.
8. SHAP.

## 15. Verificación del diseño

### 15.1 Contra la consigna
| Requisito | Dónde se cumple |
|---|---|
| ≥ 2 métodos no usados por los autores, justificados | §2.1 (KNN y MLP en ambos) |
| Valores faltantes | §0 y notebooks 01/03, sección 2 |
| Selección de *features* | §6 |
| Ajuste de hiperparámetros | §8 y §2.2 (`GridSearchCV` en las referencias) |
| Otros métodos del curso | Pipeline/ColumnTransformer, *one-hot*/ordinal, escalado, SMOTE→SMOTENC, Lasso, `GridSearchCV`, k-fold, IQR/Isolation Forest, matriz de confusión, ROC-AUC, árbol y RF, curvas de sesgo y varianza |
| Comparativa final | §10 |
| Estrategia (a) y (b) | §8: Optuna en KNN (a) y en el MLP (b) |
| Código con autoría e interpretación | §11 y §12; celdas *Qué hacemos / Por qué / Qué observamos* |

### 15.2 Contra la matriz de cobertura (§10.1 del análisis)
Las 15 filas quedan cubiertas. Cambios respecto de la matriz: SMOTE → **SMOTENC** (limitación observada: variables nominales); Boruta se descartó por su costo dentro de la CV; la selección L1 se completa con un barrido de `C` (v1.5); las curvas de aprendizaje entran al mínimo; la partición agrupada en obesidad es nueva (hallazgo del 22/11).

### 15.3 Contra las trampas documentadas
| Trampa | Tratamiento |
|---|---|
| Obesidad: target derivado de Height/Weight | Escenarios E1/E2; E2 principal |
| Obesidad: 77% sintético | Redondeo, partición agrupada, evaluación en filas reales y declaración en Discusión |
| Obesidad: duplicados | Se eliminan 24 antes de dividir |
| Obesidad: ordinales con decimales | Redondeo a la escala de la encuesta |
| Obesidad: artefacto de género | Confirmado en el EDA; revisado en la importancia por permutación (Gender, 2.ª variable); ética |
| Obesidad: menores de edad | Se conservan (113 filas) y se discute en ética la validez del IMC adulto |
| Estudiantes: nominales como enteros | *One-hot* con agrupación de niveles raros |
| Estudiantes: fuga temporal | Escenarios E1/E2; E1 principal |
| Estudiantes: Enrolled difícil | macro-F1 y F1 por clase |
| Estudiantes: macroeconómicas como cohorte | Confirmado (10 combinaciones); revisado en la interpretación |
| Estudiantes: desbalance | SMOTENC dentro del *fold* vs. sin balanceo |
| Estudiantes: momento de Debtor/Tuition (nuevo) | Limitación declarada en la Discusión (Tuition fees, 2.ª variable más importante) |

## Referencias nuevas que introduce este diseño (para verificar en la Fase 4)
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357. (Incluye SMOTE-NC).
- Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization. *Advances in Neural Information Processing Systems, 24*. (TPE) [VERIFICAR]
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. [VERIFICAR]
- Cohen, J. (1968). Weighted kappa. *Psychological Bulletin, 70*(4), 213–220. [VERIFICAR]
