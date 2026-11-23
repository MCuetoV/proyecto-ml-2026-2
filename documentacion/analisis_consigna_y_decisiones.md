# Proyecto ML — Análisis de la consigna (Paso 1: entendimiento)

**Fecha:** 17/11/2026 · **Actualizado:** 22/11/2026 (Fase 0: métodos de Martins et al. confirmados; trabajos previos del grupo de obesidad; secciones 2, 10.2, 11 y Fuentes)
**Curso:** Machine Learning (650064) — ULima, 2026-2
**Entrega:** viernes 27/11/2026
**Estado:** Fase 0 cerrada (22/11). Las decisiones de diseño se fijan en `diseno_experimental.md` (Fase 1).

---

## 0. Resumen ejecutivo

- Los **dos datasets tienen trampas metodológicas** que, si no se tratan, dan resultados inflados:
  - **Obesidad:** el target se calcula con Height y Weight (IMC), y el 77% de los datos son sintéticos (SMOTE).
  - **Estudiantes:** fuga temporal por las notas de semestre, categóricas codificadas como enteros y desbalance.
- **El 80% de la nota es el reporte.** El trabajo se planifica para el reporte y el código lo alimenta.
- Los métodos "no considerados por los autores" que sirven para **ambos** datasets incluyen **MLP, KNN, Extra Trees, AdaBoost, LDA/QDA, regresión ordinal, stacking y TabNet**.
- **Escenarios clave a evaluar:**
  - Obesidad: con Height/Weight vs. sin ellos.
  - Estudiantes: "al matricularse" vs. "al final del 2.º semestre".
- **Cruce con el curso (sección 10):**
  - Los dos métodos **vistos en clase** que ningún autor usó, en ninguno de los dos datasets, son **MLP y KNN**.
  - Además, en obesidad están permitidos **SVM, Random Forest y XGBoost** (vistos en clase, pero solo usados por los autores de estudiantes).
  - ⚠️ *Matiz del 22/11:* el grupo de autores de obesidad usó MLP, RF y SVM en otro dataset (Molina et al., 2021), y AdaBoost es frontera en estudiantes. Ver sección 2.
  - **Optuna y los algoritmos genéticos no se vieron en clase.** Son la extensión que la consigna exige; GridSearchCV (sí visto) sirve como punto de comparación.

---

## 1. Qué pide la consigna

### Requisitos explícitos
1. Dos datasets: (a) Predict Students' Dropout and Academic Success (UCI 697); (b) Estimation of Obesity Levels (UCI 544).
2. En cada dataset, **al menos 2 métodos de clasificación o regresión que los autores no usaron**, cada uno con su justificación.
3. Revisar valores faltantes.
4. Selección de features.
5. Ajuste de hiperparámetros.
6. "Otros métodos usados en el curso". Es abierto: depende de lo que realmente se dictó.
7. Comparativa final de los modelos.
8. Al menos una de estas estrategias: (a) elegir hiperparámetros con algoritmos genéticos u Optuna; (b) afinar redes neuronales con algoritmos genéticos u Optuna.
9. **Código** (repositorio o notebook): comentarios de qué hizo cada integrante, arriba o dentro de cada función, y celdas de texto que interpreten los resultados.
10. **Reporte:**
    - Título y Autores.
    - Introducción (máx. 1 pág.).
    - Antecedentes (máx. 2 págs., con libros o journals).
    - Metodología (½–1 pág., con un gráfico de los pasos).
    - Experimentación (máx. 3 págs.; empieza describiendo el dataset).
    - Resultados (~2 págs.).
    - Discusión (1–2 págs.; **obligatorio:** un par de párrafos sobre repercusiones éticas y sociales).
    - Conclusiones.
    - Aportes por integrante.
    - Referencias (APA o numérico).
11. Máximo 4 integrantes. Programa: 20%; reporte: 80%. Nota grupal.

### Requisitos implícitos (aquí se decide la nota)
- **Justificar un método** significa explicar por qué encaja con *ese* problema (tipo de variables, desbalance, target ordinal). No basta con decir que es popular.
- **Comparativa válida:** mismas particiones, mismas métricas, una línea base y una medida de variabilidad.
- **Peso 20/80:** un notebook correcto con un reporte flojo saca menos que uno modesto con un reporte riguroso.

---

## 2. Quiénes son "los autores" y qué métodos quedan descartados

### Obesidad
- El paper del enlace (Mendoza Palechor & De la Hoz Manotas, 2019, *Data in Brief* 25:104344) es un **data descriptor**: no entrena modelos. Remite al artículo completo *"Obesity Level Estimation Software based on Decision Trees"* (De-La-Hoz-Correa et al., 2019).
- Ese trabajo (*Journal of Computer Science*, 15(1), 67–77, https://doi.org/10.3844/jcssp.2019.67.77) usó la metodología SEMMA con **J48 (árbol de decisión), Naive Bayes y regresión logística (Simple Logistic)**. J48 fue el mejor, con ~97.4% de precisión. ✅ Confirmado el 22/11 por las palabras clave del registro del repositorio institucional y el resumen de la revista.
- → **Descartados:** árboles de decisión, Naive Bayes y regresión logística.
- ⚠️ **Hallazgo del 22/11:** el mismo grupo (Molina, De-La-Hoz y Mendoza, 2021, *Journal of Theoretical and Applied Information Technology*, 99(11)) entrenó **LMT, Random Forest, MLP y SVM (SMO)**, con selección de atributos (InfoGain, GainRatio, chi² y Relief) y CV de 10 *folds*. Usó **otro conjunto de datos**: 178 estudiantes reales de 18 a 25 años, sin SMOTE. No son "los autores" del dataset de UCI ni de su artículo asociado, pero un evaluador podría objetar MLP, RF o SVM como "ya usados por los autores". → Declararlo con transparencia en Antecedentes y preferir métodos que tampoco use ese grupo.
- Cañas Cervantes y Martínez Palacio (2020, *Informatics in Medicine Unlocked*, 21, 100472) usaron K-means, árboles de decisión y SVM sobre este dataset. No son los autores; sirve para Antecedentes.

### Estudiantes
- **Martins et al. (2021)**, *"Early Prediction of Student's Performance in Higher Education: A Case Study"* (WorldCIST), es el paper introductorio en UCI. Sus palabras clave incluyen clases desbalanceadas, clasificación multiclase y métodos de boosting.
  - ✅ **Confirmado el 22/11** (el texto completo es de pago; fuente secundaria que describe el procedimiento, coherente con el resumen y la lista de referencias del original: estudio comparativo en *Discover Artificial Intelligence*, 2024, https://doi.org/10.1007/s44163-023-00079-z, autores [VERIFICAR]):
    - **DT, SVM, LR y RF**, primero sin ajuste y luego con *grid search*.
    - **SMOTE vs. ADASYN**: SMOTE fue mejor; los cuatro modelos se reentrenaron con SMOTE.
    - **Boosting:** Gradient Boosting, XGBoost, CatBoost y **LogitBoost** (no LightGBM), ajustados con búsqueda aleatoria. XGBoost fue el mejor, muy cerca de Gradient Boosting.
    - Métrica: **F1 promedio de las tres clases** para el ajuste y accuracy como métrica global.
    - **Datos: 3623 registros y 25 variables, solo información disponible al momento de la matrícula.** Las notas del primer año figuran como trabajo futuro. La versión de UCI (4424 × 36) es una ampliación posterior. → El escenario "al matricularse" es el único comparable con los autores.
  - **LogitBoost** proviene de la relectura de AdaBoost como regresión logística aditiva (Friedman et al., 2000, citado por los autores). → **AdaBoost es un método "frontera" en estudiantes.**
  - El estudio de 2024 ya usó Optuna y SHAP sobre este dataset: citarlo en Antecedentes.
- **Realinho et al. (2022)**, *Data* 7(11):146, es el descriptor. Menciona RF, XGBoost, LightGBM y CatBoost solo en el análisis de importancia de features por permutación, no como modelos reportados. Cita SMOTE, ADASYN, Balanced RF y Easy Ensemble como opciones, pero no dice haberlos aplicado.
- → **Descartados:** LR, SVM, DT, RF y toda la familia de gradient boosting.

### Consecuencia
- Lo que nadie usó: **MLP (red neuronal), KNN, Extra Trees, AdaBoost, LDA/QDA, regresión ordinal, stacking y TabNet**. Matices del 22/11: MLP, RF y SVM sí los usó el grupo de obesidad con otro dataset (Molina et al., 2021), y AdaBoost es frontera en estudiantes (LogitBoost). **KNN es el único método del curso limpio en todos los casos.**
- Si se eligen **los mismos 2–3 métodos nuevos para ambos datasets**, la justificación y la comparativa salen más limpias, y el trabajo se reparte mejor.
- ⚠️ Evitar métodos "frontera" como HistGradientBoosting: sigue siendo gradient boosting y un evaluador podría objetarlo.

---

## 3. Dataset de estudiantes (UCI 697): hechos y trampas

### Hechos
- 4424 registros (estudiantes), 36 features más el target. **Sin valores faltantes.**
- Target de 3 clases: **Graduate 2209 (50%) · Dropout 1421 (32%) · Enrolled 794 (18%)**. El estado se mide al final de la duración normal de la carrera.
- Grupos de variables:
  - Demográficas (6).
  - Socioeconómicas (8).
  - Macroeconómicas (3: desempleo, inflación, PIB).
  - Académicas al momento de la matrícula (5).
  - Académicas al final del 1.º y 2.º semestre (6 + 6).
- Los autores usaron una división de 80% para entrenamiento y 20% para test.
- Datos del Instituto Politécnico de Portalegre (Portugal). Licencia CC BY 4.0.

### Trampas
1. **Variables categóricas guardadas como enteros.** Estado civil, modo de postulación, carrera, nacionalidad y calificación/ocupación de los padres aparecen como Integer, pero son nominales con muchos niveles. Si entran como números, KNN y MLP aprenden un orden falso. Hay que elegir y justificar la codificación (one-hot, agrupar niveles poco frecuentes, target encoding).
2. **Fuga temporal según el caso de uso.** Las notas de 1.º y 2.º semestre dominan la predicción. Para una alerta temprana, al momento de la matrícula esas variables no existen. → **Comparar dos escenarios: "al matricularse" vs. "al final del 2.º semestre".**
3. **La clase "Enrolled" es intrínsecamente difícil.** Es un estado intermedio (aún puede terminar en graduación o retiro), es la minoritaria y la que peor se predice. → No usar accuracy como métrica principal: usar **macro-F1 o accuracy balanceado**, con F1 por clase.
4. **Variables macroeconómicas.** Toman pocos valores distintos, uno por cohorte: funcionan como un "año de ingreso" disfrazado. Hay que interpretarlas con cuidado.
5. **Desbalance.** Si se usa SMOTE u otro remuestreo, se aplica **solo dentro de los folds de entrenamiento** (pipeline de imblearn), nunca antes de dividir.
6. **Atributos sensibles** para la sección de ética: género, nacionalidad, estado civil, edad, deudor, becario, necesidades educativas especiales.

---

## 4. Dataset de obesidad (UCI 544): el más delicado

### Hechos
- 2111 registros, 16 features más el target (`NObeyesdad`). **Sin valores faltantes.**
- Encuesta web anónima, abierta 30 días, en Barranquilla, Lima y Ciudad de México. Edades de 14 a 61 años.
- **485 registros reales (~23%). El 77% es sintético**, generado con SMOTE en Weka.
- Atributos:
  - Gender, Age, Height (m), Weight (kg), family_history_with_overweight.
  - FAVC: comida hipercalórica.
  - FCVC: verduras (nunca / a veces / siempre).
  - NCP: número de comidas principales.
  - CAEC: come entre comidas.
  - SMOKE.
  - CH2O: agua.
  - SCC: monitorea calorías.
  - FAF: actividad física.
  - TUE: tiempo en dispositivos.
  - CALC: alcohol.
  - MTRANS: medio de transporte.
- Target ordinal de 7 clases: Insufficient Weight, Normal Weight, Overweight I, Overweight II, Obesity I, Obesity II, Obesity III. Tras SMOTE quedan aproximadamente balanceadas (~270–350 por clase; **verificar en el EDA**).
- Licencia CC BY 4.0.

### Trampas (de mayor a menor gravedad)
1. **El target se calcula a partir de dos features.** La etiqueta sale de IMC = peso / altura² con los umbrales de la OMS y la NOM-008-SSA3-2010. Con Height y Weight dentro del modelo, la tarea se reduce a reconstruir el IMC: por eso cualquier modelo se acerca a ~97%.
   → **Evaluar dos escenarios: con Height/Weight y sin ellos (solo hábitos y condición física).** El segundo responde la pregunta que el título del dataset promete. Es probablemente la decisión que más valor le da al reporte.
2. **Los datos sintéticos inflan la evaluación.** SMOTE se aplicó antes de cualquier partición, así que el test contiene "casi copias" de filas del train. Los propios autores advierten que SMOTE puede propagar ruido.
   - Hay que declararlo como limitación.
   - **Pista útil:** las respuestas reales de la encuesta son enteras y las sintéticas tienen decimales en variables ordinales (FCVC, NCP, FAF, TUE). Esto permite **aproximar cuáles filas son reales**, para el EDA o para una evaluación más honesta.
3. **Variables ordinales convertidas en continuas** (por ejemplo, FCVC = 2.37). Hay que decidir si se redondean o se dejan continuas, y justificarlo.
4. **Artefactos demográficos del sobremuestreo.** Hallazgo conocido, **por verificar en el EDA**: Obesity Type II es casi solo masculino y Type III casi solo femenino. Si se confirma, un modelo puede predecir el tipo de obesidad por el sexo. Es un sesgo espurio y un tema directo para ética.
5. **Etiqueta poco clara.** Los umbrales dan un solo rango de sobrepeso (25.0–29.9), pero la etiqueta tiene dos niveles de sobrepeso y el paper no explica cómo se separan. Es relevante si se plantea como regresión del IMC y luego se mapea a clases.
6. **El target es ordinal.** Confundir "Normal" con "Obesity III" es peor que confundir "Overweight I" con "Overweight II". Esto justifica métodos ordinales o métricas como el **kappa ponderado cuadrático** o el MAE entre clases, y conecta con el "clasificación o regresión" de la consigna.
7. **Otras limitaciones:** datos autorreportados; solo tres ciudades; incluye **menores de edad** (desde 14 años), y el IMC adulto no se aplica directamente a adolescentes. Es un punto ético y de validez.

---

## 5. Ambigüedades para consultar con el profesor

1. **Límites de páginas.** La plantilla dice "el dataset seleccionado", en singular; parece reciclada de una versión con un solo dataset. Con dos datasets, 3 páginas de Experimentación y 2 de Resultados quedan muy ajustadas. ¿Los límites son por dataset o totales?
2. **"Los autores".** En obesidad, ¿cuentan los del descriptor (que no entrenan modelos) o los del artículo completo (J48, NB, LR)? La lectura propuesta es el artículo completo.
3. **"Otros métodos usados en el curso".** ✅ Resuelto con el levantamiento del material del 18/11 (secciones 9 y 10). Falta incorporar lo que se dicte el 18/11, el 23/11 y el 25/11 (según el sílabo, probablemente boosting y Bayes).
4. **Estrategia (a) vs. (b).** Piden "alguno". Se pueden cubrir las dos con poco costo extra: si el MLP es uno de los métodos nuevos, ajustarlo con Optuna cumple (b), y Optuna para el resto cumple (a). Optuna incluye además un sampler genético (NSGA-II), lo que permite mencionar ambos enfoques con fundamento.
5. **Formato del código.** ¿Un notebook por dataset o uno solo? ¿Repositorio?

---

## 6. Principios de rigor (las reglas del juego, no soluciones todavía)

- **Test de reserva fijo y estratificado**, que no se toca hasta el final. Todo el ajuste se hace con CV dentro del train.
- **Todo preprocesamiento dentro del pipeline:** escalado, codificación, remuestreo y selección de features. Si se seleccionan features fuera de la CV, el error sale optimista.
- **Líneas base:** un clasificador trivial (DummyClassifier) y **reproducir un método de los autores** como referencia, para comparar en igualdad de condiciones y no contra números de papers con otro protocolo.
- **Métricas:**
  - Estudiantes: macro-F1, F1 por clase y accuracy balanceado.
  - Obesidad: macro-F1 y kappa ponderado o una métrica ordinal.
- **Variabilidad:** media ± desviación estándar con CV repetida. Opcional: una prueba estadística simple (Wilcoxon o Friedman) para decir si las diferencias entre modelos son reales.
- **Reproducibilidad:** semillas fijas y entorno documentado.

---

## 7. Restricción de tiempo y equipo

- Entrega el **viernes 27/11/2026**: ~10 días desde el 17/11, con un equipo de hasta 4 integrantes.
- **Alcance acotado:**
  - 2–3 métodos nuevos, comunes a ambos datasets.
  - 1–2 líneas base.
  - Optuna con presupuesto de trials limitado.
  - Los dos escenarios clave por dataset.
  - No hacen falta diez modelos.
- **Escribir el reporte en paralelo al código.** Introducción, Antecedentes y Ética se pueden redactar desde el inicio.
- **Repartir pensando en la sección de Aportes:** cada integrante con una parte de código y una de reporte claramente identificables.

---

## 8. Siguiente paso (próximo chat)

Fijar las decisiones de diseño:
1. Qué métodos nuevos usar y su justificación.
2. Escenarios por dataset.
3. Métricas y protocolo de validación.
4. Qué técnicas "del curso" se aplican.
5. Reparto del trabajo entre los 4 integrantes.
6. Estructura del repositorio o notebooks.
7. Esqueleto del reporte con presupuesto de páginas.

**Preguntas abiertas:**
- ¿Ya están definidos los 4 integrantes?
- ¿Un notebook por dataset o uno solo?
- ¿Las respuestas del profesor a la sección 5?
- ¿El MLP se implementa en PyTorch (como en clase) o con `MLPClassifier` de scikit-learn? Ver la sección 10.3.

---

## 9. Inventario del material del curso (levantado el 18/11/2026)

**Fuente:** carpeta local `CURSOS_UNIVERSIDAD/MachineLearning` (sílabo, PPT, notebooks y Excel).
**Cobertura:** clases hasta el miércoles 11/11 (semana 12). El lunes 16/11 no hubo clase.
**Calendario:** semana 1 = 24/08; E1 en la semana 5; E2 el lunes 26/10 (semana 10).

> El orden real de las clases no siguió el del sílabo. Por ejemplo, redes neuronales se dictó en el segundo bloque, aunque los archivos se llamen "Semana 01". El inventario se organiza por bloque y tema.

### Bloque 1 — Hasta el E1 (semanas 1–5)

| Tema | Archivos | Técnicas y herramientas vistas |
|---|---|---|
| Conceptos de ML y tipos de datos | `SEMANA 01- Conceptos Introductorios y manejo de datos.pptx` | Definición de ML (Mitchell); variables nominales, ordinales y numéricas; flujo de trabajo de ML; datos estructurados y no estructurados |
| EDA completo (caso de fuga de clientes) | `Semana_01_ML_EDA.ipynb` | Nulos, duplicados, análisis univariante y multivariante, correlación numérica y categórica (chi²), LabelEncoder, BinaryEncoder, MinMax y StandardScaler, Decision Tree y RF como modelos de prueba, **división train/valid/test**, **submuestreo y sobremuestreo aplicados solo al train** (RandomOver, ClusterCentroids, SMOTE, SMOTE-Tomek, ADASYN) |
| Imputación | `SESION_01_EDA_IMPUTACIÓN_DE_DATOS.ipynb` (+ PDF) | `missingno`, forward/backward fill, SimpleImputer, **KNNImputer**, IterativeImputer (ExtraTrees), eliminación de duplicados; KNeighborsClassifier |
| Escalamiento y encoding | `Semana_01_EDA_ESCALAMIENTO_Y_ENCODING.ipynb` | OneHot, Ordinal, Label y Binary encoder; StandardScaler, MinMaxScaler, PowerTransformer; **ColumnTransformer y Pipeline**; t-SNE |
| Outliers | `SEMANA 02-03(1).pptx`, `Semana_03_ML_EDA_TRATAMIENTO_DE_OUTLIERS.ipynb`, `Algoritmos_detección_outliers.ipynb` | IQR, Z-score, **LOF** (programado desde cero), **Isolation Forest**, PyOD (KNN, ABOD, CBLOF, HBOS, ECOD, PCA), **KFold con cross_val_score** |
| Selección de variables y reducción de dimensionalidad | `SEMANA 03_Seleccion de Variables(1)(1).pptx`, `SEMANA_03_EDA_SELECCIÓN_DE_CARACTERISTICAS.ipynb` | Métodos filter (**chi²**, VarianceThreshold), wrapper (**stepwise por p-values**), embebidos (**Lasso**, importancia de RF con SelectFromModel), **Boruta**, **SHAP**, **PCA**, MCA y FAMD; split estratificado |
| Regresión lineal | `semana4.pptx`, `regresion_metricas_avanzado.pptx`, `SEMANA_04_ML_1_REGRESION_LINEAL.ipynb` | OLS (sklearn y statsmodels), ecuación normal, gradient descent, los 5 supuestos (Durbin-Watson, Breusch-Pagan, Shapiro, VIF), R², MAE, MSE, RMSE, MAPE |
| Regularización | `SEMANA_04_ML_2_REGRESION_REGULARIZACIÓN.ipynb` | Ridge, Lasso, Elastic Net; **GridSearchCV** para elegir alpha |

### Bloque 2 — Del E1 al E2 (semanas 6–9)

| Tema | Archivos | Técnicas y herramientas vistas |
|---|---|---|
| Regresión logística y MLE | `semana6.pptx`, `LOG-LIKELIHOOD(1).xlsx` | Sigmoide, likelihood y log-likelihood, BCE, **One-vs-All y softmax** (multiclase); bias y varianza, under/overfitting, **k-fold CV**, Ridge/Lasso en logística (C = 1/λ) |
| Evaluación de modelos | `SEMANA 08 - Evaluación de Modelos.pptx`, `semana8.pptx` (respaldo 2025-1) | Matriz de confusión, accuracy, **precision, recall y F1 por clase**, **curva ROC y AUC**, métricas de regresión (incluye R² ajustado y Huber) |
| Redes neuronales: fundamentos y entrenamiento | `SEMANA 01- Fundamentos de Redes Neuronales.pptx`, `SEMANA 01_02 - Entrenamiento de Redes Neuronales.pptx`, `Semana_01_Fundamentos_de_Redes_Neuronales.ipynb`, `semana_1_2_RedesNeuronales.ipynb` | Perceptrón, activaciones (sigmoide, tanh, ReLU), descenso del gradiente, batch / mini-batch / SGD, BCE y CCE, backpropagation; **optimizadores SGD, Momentum, Adagrad, RMSprop, Adam y AdamW**; **MLP en PyTorch** (MNIST, variando capas, activación y optimizador) |
| Tipos de redes | `SEMANA 02- Redes neuronales.pptx`, `SEMANA_7_REDES_RECURRENTES.ipynb` | **Datos tabulares → MLP**; CNN; RNN, LSTM y GRU (PyTorch) |
| Material propio para el E2 | `ML_E2_indice_material.md`, `Apuntes_E2_...pdf`, `Practica_E2_ML.xlsx`, `Ejercicios_...pdf` | Índice y plantillas de los bloques 1–2 |

### Bloque 3 — Después del E2 (semanas 10–12)

| Tema | Archivos | Técnicas y herramientas vistas |
|---|---|---|
| SVM | `SVM.ipynb` | SVC lineal y RBF, kernel trick (visualización 3D), **GridSearchCV (C, gamma; cv=5)**, SVR lineal y RBF, **OneVsRestClassifier** para multiclase |
| Balanceo de datos | `Semana_10_Balance_de_datos.ipynb` | SMOTE (algoritmo paso a paso), Borderline-SMOTE, SMOTE + Tomek Links, ADASYN, RandomUnder/Over, ClusterCentroids; **remuestreo solo sobre X_train**; evaluación con XGBClassifier y `classification_report` |
| Árboles y Random Forest | `ARBOL_REGRESION.ipynb`, `ARBOL_REGRESION_(2).ipynb`, `ARBOL.xlsx`, `EJEMPLO ARBOL CLASIFICACION.xlsx` | Árbol de regresión **programado desde cero** (MSE ponderado, `min_points_leaf`, `max_num_leaf`), extensión a N features, **Random Forest desde cero** (bootstrap de filas y submuestreo de columnas), RMSE; árbol de clasificación con **Gini** en Excel |

### Pendiente o no encontrado en el material

- **Por dictar según el sílabo (semanas 12–15):** boosting (AdaBoost, XGBoost, LightGBM), probabilidad y Bayes, clustering (K-means, DBSCAN, GMM) y NLP. Puede caer en las clases del 18/11, 23/11 o 25/11.
- **Del sílabo, sin rastro en el material:** curvas de error, **nested cross-validation**, CV para series temporales, **data leakage** (como tema explícito) y KNN como clasificador principal. El KNN solo aparece como imputador, en PyOD y como modelo de prueba.
- **Nunca usado en el curso:** Optuna, algoritmos genéticos, RandomizedSearchCV, StratifiedKFold, `imblearn.pipeline`, `f1_score` macro, balanced accuracy, kappa, `learning_curve`, `MLPClassifier` de scikit-learn, Keras.

### Herramientas del curso

pandas, numpy, scikit-learn, imbalanced-learn, PyTorch, XGBoost, statsmodels, scipy, PyOD, category_encoders, Boruta, SHAP, prince, missingno, seaborn, matplotlib y plotly. Entorno local: Python 3.13 con `requirements.txt` en la carpeta del curso.

### Referencias oficiales del sílabo (para Antecedentes)

- Géron, A. (2023). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.
- Murphy, K. P. (2012). *Machine learning: A probabilistic perspective*. MIT Press.
- Tan, P.-N., Steinbach, M., & Kumar, V. (2018). *Introduction to data mining* (2nd ed.). Pearson.

---

## 10. Matriz de cobertura: consigna × curso × brecha × extensión

**Regla aplicada:** en cada etapa partimos de lo visto en clase y solo lo superamos cuando los datos muestran una limitación concreta de esa técnica. Cada extensión queda justificada por un problema documentado en el reporte.

### 10.1 Requisitos de la consigna

| # | Requisito | Lo que da el curso | Brecha en este proyecto | Extensión propuesta y por qué |
|---|---|---|---|---|
| 1 | Valores faltantes | `missingno`, SimpleImputer, KNNImputer, IterativeImputer, duplicados | Ninguno de los dos datasets tiene nulos | Documentar el chequeo con las herramientas del curso. Revisar **duplicados** (obesidad) y la **huella de SMOTE**: decimales en variables ordinales, que permiten aproximar las ~485 filas reales |
| 2 | EDA y outliers | Univariante y multivariante, chi², IQR, Z-score, LOF, Isolation Forest | En obesidad los "outliers" pueden ser artefactos sintéticos; en estudiantes, notas 0 que probablemente son legítimas (unidades sin evaluación; verificar en el EDA) | Usar LOF e Isolation Forest **solo como diagnóstico**, sin eliminar filas a ciegas; justificar caso por caso |
| 3 | Preprocesamiento (encoding y escalado) | One-hot, Ordinal, Binary encoder; Standard/MinMax/PowerTransformer; ColumnTransformer + Pipeline | Estudiantes: nominales con muchos niveles codificadas como enteros. Obesidad: ordinales con decimales | One-hot o BinaryEncoder (del curso) para nominales; agrupar niveles poco frecuentes; decidir el redondeo de ordinales. **Todo dentro de un ColumnTransformer en el Pipeline** |
| 4 | Selección de features | VarianceThreshold, chi², stepwise, Lasso, SelectFromModel con RF, Boruta, SHAP, PCA, MCA, FAMD | En clase la selección se hizo sobre un solo split, fuera de la validación cruzada | Aplicar 1–2 métodos del curso (p. ej. Lasso/L1 y Boruta o importancia de RF) **dentro del Pipeline evaluado con CV**, para evitar sesgo de selección. Comparar el modelo con todas las variables vs. el seleccionado |
| 5 | Desbalance | SMOTE, Borderline, SMOTE-Tomek, ADASYN, under-sampling; solo sobre el train | Con CV hay que remuestrear en cada fold, no una sola vez | `imblearn.pipeline.Pipeline` para remuestrear **dentro de cada fold**; comparar con `class_weight='balanced'`. **En obesidad no se remuestrea** (ya fue balanceado con SMOTE) |
| 6 | ≥ 2 métodos que los autores no usaron | MLP (PyTorch), KNN, SVM, árboles, RF, XGBoost, logística, regresión lineal y regularizada | Ver la tabla 10.2 | Núcleo común: **MLP + KNN** (vistos en clase y no usados por ningún autor). Opcionales por dataset en la tabla 10.2 |
| 7 | Ajuste de hiperparámetros | GridSearchCV (cv=5) para Ridge/Lasso y SVM | No se vio búsqueda eficiente ni bayesiana | **Optuna (TPE)** como estrategia principal. GridSearchCV como línea base del curso, para comparar calidad y costo (n.º de evaluaciones y tiempo) |
| 8 | Estrategia (a) o (b): AG / Optuna | Ni Optuna ni algoritmos genéticos | Requisito totalmente nuevo | (a) Optuna para KNN y el resto de modelos. (b) Optuna para el **MLP**: capas, neuronas, activación, optimizador (Adam/AdamW/SGD, vistos en clase), learning rate, weight decay y batch. Opcional: el sampler **NSGA-II** de Optuna (genético) para cubrir ambos enfoques |
| 9 | Protocolo de validación | train/test, train/valid/test, KFold, GridSearch con cv=5 | Sin estratificación, sin CV repetida y sin nested CV (que figura en el sílabo, pero no en el material) | Test de reserva estratificado (80/20, como los autores de estudiantes) + **StratifiedKFold repetido** dentro del train para el ajuste. Nested CV solo si el tiempo alcanza |
| 10 | Métricas | Matriz de confusión, accuracy, precision, recall, F1 por clase, ROC/AUC; MAE, RMSE, R² | Accuracy engaña con desbalance (estudiantes) y con target ordinal (obesidad) | Reportar las del curso y agregar **macro-F1**, **balanced accuracy**, ROC-AUC OvR multiclase y, en obesidad, **kappa ponderado cuadrático** |
| 11 | Overfitting, bias y varianza | Teoría de bias/varianza, under/overfitting, regularización, CV | No se graficaron curvas | **Curvas de aprendizaje y de validación** del mejor modelo (sustentan la Discusión) |
| 12 | Interpretabilidad | SHAP, importancia de RF | — | **SHAP o permutation importance** sobre el mejor modelo; Realinho et al. usaron permutation importance, así que permite contrastar |
| 13 | Comparativa final | Tablas de métricas por modelo | Sin variabilidad ni prueba estadística | Media ± desviación por CV, **DummyClassifier** como piso, **un método de los autores reproducido** como referencia y, opcionalmente, una prueba de Wilcoxon o Friedman |
| 14 | Multiclase | One-vs-All, softmax, OneVsRestClassifier | Ambos datasets son multiclase (3 y 7 clases) | Usar el manejo nativo de cada modelo; mencionar OvR / softmax en Antecedentes |
| 15 | Reporte: Antecedentes | Géron, Murphy, Tan (sílabo) | — | Citar las referencias del sílabo más los papers de cada técnica nueva (Optuna: Akiba et al., 2019; SMOTE: Chawla et al., 2002) |

### 10.2 Métodos candidatos: curso × autores

| Método | ¿Visto en clase? | Autores obesidad (J48, NB, LR) | Autores estudiantes (LR, SVM, DT, RF, boosting)¹ | Apto en obesidad | Apto en estudiantes |
|---|---|---|---|---|---|
| **MLP (red neuronal)** | Sí (PyTorch) | No usado² | No usado | ✅² | ✅ |
| **KNN** | Parcial (imputador, prueba, sílabo sem. 2/7) | No usado | No usado | ✅ | ✅ |
| SVM | Sí (sem. 10–11) | No usado² | Usado | ✅² | ❌ |
| Random Forest | Sí (desde cero) | No usado² | Usado | ✅² | ❌ |
| XGBoost | Usado en notebooks; teoría pendiente (sem. 12) | No usado | Usado | ✅ | ❌ |
| AdaBoost | Pendiente (sem. 12) | No usado | Frontera¹ | ✅ si se dicta | ⚠️ frontera |
| Naive Bayes | Pendiente (sem. 13) | Usado | No usado | ❌ | ✅ si se dicta |
| Ridge / Lasso / Elastic Net | Sí (regresión) | Logística sí | Logística sí | Solo como regresión del IMC | ❌ |
| Árbol de decisión | Sí | Usado | Usado | ❌ | ❌ |
| Regresión logística | Sí | Usado | Usado | ❌ (solo como reproducción de los autores) | ❌ (ídem) |

¹ Martins et al. (2021) usaron LogitBoost, variante de AdaBoost (confirmado el 22/11; ver sección 2).
² Usado por el mismo grupo de autores con otro dataset de 178 registros (Molina et al., 2021). Defendible, pero se declara.

**Lectura:**
- **MLP y KNN** cumplen el requisito en los dos datasets con material del curso, y se justifican por sí solos.
  - MLP: el PPT de tipos de redes indica *datos tabulares → MLP*, y su ajuste fino es la estrategia (b).
  - KNN: modelo basado en distancias, que contrasta con los modelos de árboles y lineales de los autores. Exige un buen encoding y escalado, que son temas del curso.
- **Tercer método opcional:** SVM o RF en obesidad (ya vistos en clase), o AdaBoost en ambos si se dicta a tiempo. *Revisado el 22/11: SVM y RF quedan debilitados en obesidad (nota ²) y AdaBoost en estudiantes (nota ¹); se decide en el documento de diseño.*
- **Para regresión en obesidad (opcional):** predecir el IMC con SVR, MLP o Elastic Net, mapearlo a las 7 clases y comparar. Conecta con el "clasificación o regresión" de la consigna y con el target ordinal.

### 10.3 Decisión pendiente de implementación: ¿MLP en PyTorch o en scikit-learn?

| | PyTorch (como en clase) | `MLPClassifier` (scikit-learn) |
|---|---|---|
| Coherencia con el curso | Alta: mismo framework y optimizadores (Adam, AdamW, SGD) | Media: mismo concepto, otra herramienta |
| Integración con Pipeline, CV y Optuna | Manual (bucle de entrenamiento, early stopping, escalado) | Directa |
| Hiperparámetros ajustables | Todos (dropout, AdamW, scheduler) | Capas, activación, `solver` (adam/sgd), alpha (L2), learning rate y batch |
| Costo en tiempo | Mayor | Menor |

**Recomendación preliminar:** `MLPClassifier` para el pipeline comparativo, citando en Antecedentes la teoría vista en clase. PyTorch solo si sobra tiempo o si el profesor lo valora explícitamente. **A decidir en el siguiente paso.**

---

## 11. Registro de decisiones (insumo para la declaración de uso de IA)

| Fecha | Decisión |
|---|---|
| 17/11 | Fecha de entrega: **viernes 27/11/2026**. |
| 17/11 | Uso de IA **validado por el profesor**. La declaración de uso de IA va **al final del reporte**, redactada cuando se conozca todo lo realizado. Este registro se mantiene en cada paso para escribirla. |
| 17/11 | Principio de trabajo: **base del curso + extensión justificada** por una limitación observada en los datos. Toda extensión debe poder explicarla cualquier integrante. |
| 17/11 | Compromisos del equipo: citar todo lo que se tome de otros autores, entender rigurosamente el código y distribuir bien el trabajo. |
| 18/11 | Material del curso levantado con clases hasta el 11/11 (16/11 sin clase). Incorporar lo que se dicte el 18/11, 23/11 y 25/11. |
| 18/11 | Núcleo propuesto de métodos nuevos: **MLP + KNN** en ambos datasets (por confirmar en el paso de diseño). |
| 22/11 | Se recibió la plantilla oficial del reporte (`Plantilla_trabajo_ML-2.docx`). **Su diseño no se modifica**: A4, Times New Roman 12, secciones numeradas 1–7 y Referencias, encabezados de páginas pares e impares. Integrantes en la portada: Eros Bellido, Eduardo Tello y Marcelo Cueto. |
| 22/11 | Se crearon tres skills de revisión para el reporte: lenguaje en español (`revision-lenguaje-es`), APA 7 (`revision-apa7-es`) y coherencia (`revision-coherencia-informe`). Párrafos con sangría de primera línea. |
| 22/11 | Todo el código debe ser **completamente explicable** por cualquier integrante y partir de lo aprendido en el curso. |
| 22/11 | Ajustes a la plantilla (`Plantilla_trabajo_ML_v2.docx`): subtítulo "Proyecto del curso Machine Learning (650064), ciclo 2026-2"; asesor Edwin Escobedo Cárdenas; secciones nuevas **8. Aportes** y **9. Declaración de uso de inteligencia artificial**; sangría de primera línea (1,27 cm) en el cuerpo y sangría francesa en Referencias. Título y encabezados: pendientes hasta definir el nombre del trabajo. |
| 22/11 | Las referencias que trajo la plantilla (Breiman, Chen y Guestrin, Freund y Schapire, Goodfellow et al.) fueron un aporte previo de un compañero; **no implican métodos elegidos**. Se corrigieron (duplicado, autor, orden, formato APA) y se eliminarán si no se citan. |
| 22/11 | Redacción en **español de Latinoamérica** (variante peruana). Los términos técnicos propios del inglés se pueden mantener, en cursiva y explicados en su primera mención. Punto decimal. |
| 22/11 | Aún no hay decisiones de diseño (métodos, escenarios, métricas). Se elaborarán de forma rigurosa con Claude antes de escribir código. |
| 22/11 | El equipo tiene **4 integrantes**: Eros Bellido (20182315), Eduardo Tello (20212654), Marcelo Cueto (20200614) y un cuarto integrante cuyos datos están pendientes. |
| 22/11 | **Repositorio en GitHub:** `MCuetoV/proyecto-ml-2026-2` (privado; confirmar visibilidad), clonado en `CURSOS_UNIVERSIDAD/MachineLearning/proyecto-ml-2026-2`. Marcelo trabaja en VS Code y el resto en Colab conectado a GitHub, de modo que cada *commit* registra a su autor. Guía completa en `proyecto-ML/guia_trabajo_github_colab.md`. |
| 22/11 | **Organización del código:** cuatro notebooks (exploración y modelado de cada dataset), cada uno con un solo responsable, y `utilidades.py` para las funciones comunes, editado solo por Marcelo. Los notebooks gemelos siguen la misma estructura de secciones, alineada con el reporte. Las tablas y figuras del reporte las genera el código en `resultados/`. Responsables por definir. |
| 22/11 | **Fase 0 cerrada.** Métodos de Martins et al. (2021) confirmados: DT, SVM, LR, RF, Gradient Boosting, XGBoost, CatBoost y LogitBoost, con SMOTE; datos de 3623 × 25 solo de matrícula. AdaBoost pasa a método frontera en estudiantes. |
| 22/11 | Hallazgo: el grupo de autores de obesidad usó MLP, RF, SVM y LMT en otro dataset de 178 registros (Molina et al., 2021). Se declara en Antecedentes; KNN es el único método del curso limpio en ambos datasets. |
| 22/11 | Límites de páginas sin aclarar por el profesor: se asumen **totales**. Redacción concisa, sin relleno. |
| 22/11 | Datos del cuarto integrante: los agrega el equipo manualmente al final. Visibilidad del repositorio e invitaciones: a cargo de Marcelo. Título: se define al final. Sin material nuevo en la clase del 18/11. |
| 22/11 | Marcelo pide trabajar el código **usando Git correctamente**; el flujo se define en el documento de diseño. |
| 22/11 | Los CSV de UCI no se pueden descargar desde los entornos de Claude (proxy bloquea archive.ics.uci.edu): Marcelo los descarga y los coloca en `datos/`. |

**Uso de IA hasta ahora (para la declaración):** análisis de la consigna y de los papers de los datasets (17/11); revisión automatizada del material del curso para el inventario y la matriz de cobertura (18/11); análisis de la plantilla y diseño de los skills de revisión de lenguaje, APA y coherencia (22/11); verificación en fuentes de los métodos usados por los autores de ambos datasets y revisión del estado del repositorio (Fase 0, 22/11). Todo con Claude (Anthropic).

---

## Fuentes
- Mendoza Palechor, F., & De la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. *Data in Brief, 25*, 104344. https://doi.org/10.1016/j.dib.2019.104344 — PDF: https://rcastoragev2.blob.core.windows.net/92f60d2730c6f5dd7bf2775c4e89de41/PMC6710633.pdf
- De-La-Hoz-Correa, E., Mendoza Palechor, F., De-La-Hoz-Manotas, A., Morales Ortega, R., & Sánchez Hernández, A. B. (2019). Obesity level estimation software based on decision trees. *Journal of Computer Science, 15*(1), 67–77. https://doi.org/10.3844/jcssp.2019.67.77 (repositorio: https://hdl.handle.net/11323/4176)
- Molina, D., De-La-Hoz, A., & Mendoza, F. (2021). Classification and features selection method for obesity level prediction. *Journal of Theoretical and Applied Information Technology, 99*(11). https://core.ac.uk/download/475504217.pdf
- Cañas Cervantes, R., & Martínez Palacio, U. (2020). Estimation of obesity levels based on computational intelligence. *Informatics in Medicine Unlocked, 21*, 100472.
- [VERIFICAR autores] (2024). Supervised machine learning algorithms for predicting student dropout and academic success: A comparative study. *Discover Artificial Intelligence*. https://doi.org/10.1007/s44163-023-00079-z
- UCI — Estimation of Obesity Levels: https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition (DOI 10.24432/C5H31Z)
- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting Student Dropout and Academic Success. *Data, 7*(11), 146. https://doi.org/10.3390/data7110146 — PDF: https://dorsu.edu.ph/faculty/jscabrera/wp-content/uploads/2022/12/data-07-00146.pdf
- Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: A case study. En *Trends and Applications in Information Systems and Technologies* (WorldCIST 2021), pp. 166–175. https://doi.org/10.1007/978-3-030-72657-7_16
- UCI — Predict Students' Dropout and Academic Success: https://archive.ics.uci.edu/dataset/697
- Material del curso: carpeta local `CURSOS_UNIVERSIDAD/MachineLearning` y `Silabo_MachineLearnin2026-2.pdf` (referencias del sílabo en la sección 9).
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD*, 2623–2631. https://doi.org/10.1145/3292500.3330701
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357. https://doi.org/10.1613/jair.953
