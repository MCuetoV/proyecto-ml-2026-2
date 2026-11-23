"""Funciones comunes del proyecto de Machine Learning (ULima 2026-2).

Las usan los cuatro notebooks. Cada función implementa una decisión del
documento de diseño (documentacion/diseno_experimental.md); la sección del
diseño se indica en su docstring.

Uso desde un notebook (después de la celda 0):
    import utilidades
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constantes del proyecto
# ---------------------------------------------------------------------------
SEMILLA = 42                     # semilla única para todo el proyecto
SEMILLA_COMPARATIVA = 43         # semilla distinta para la CV de la comparativa (diseño, sección 7)
CARPETA_DATOS = "datos"
CARPETA_TABLAS = os.path.join("resultados", "tablas")
CARPETA_FIGURAS = os.path.join("resultados", "figuras")

# Estudiantes (UCI 697)
OBJETIVO_ESTUDIANTES = "Target"
CLASES_ESTUDIANTES = ["Dropout", "Enrolled", "Graduate"]   # códigos 0, 1 y 2
COLUMNAS_ESTUDIANTES = [   # las 36 variables de entrada, en el orden del CSV
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification', 'Previous qualification (grade)',
    'Nacionality', "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", 'Admission grade', 'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment', 'International',
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP',
]
NOMINALES_ESTUDIANTES = [
    "Marital status", "Application mode", "Course", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation",
]
BINARIAS_ESTUDIANTES = [
    "Daytime/evening attendance", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "International",
]

# Obesidad (UCI 544)
OBJETIVO_OBESIDAD = "NObeyesdad"
ORDEN_OBESIDAD = [               # orden de menor a mayor severidad: códigos 0 a 6 (para el kappa ponderado)
    "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II",
    "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
]
COLUMNAS_OBESIDAD = [      # las 16 variables de entrada, en el orden del CSV
    "Gender", "Age", "Height", "Weight", "family_history_with_overweight", "FAVC", "FCVC", "NCP",
    "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS",
]
ANTROPOMETRIA = ["Height", "Weight"]
ENCUESTA_OBESIDAD = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]   # respuestas discretas de la encuesta
BINARIAS_OBESIDAD = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
NOMINALES_OBESIDAD = ["MTRANS"]
ORDEN_CAEC = ["no", "Sometimes", "Frequently", "Always"]
ORDEN_CALC = ["no", "Sometimes", "Frequently"]                # "Always" (1 fila) se fusiona con "Frequently"
COLUMNAS_AUXILIARES = ["probablemente_real", "grupo"]         # no son variables del modelo


def cargar_estudiantes():
    """Carga el CSV de estudiantes y limpia los nombres de columnas.

    El archivo usa ";" como separador, trae BOM y una columna termina en
    tabulación ("Daytime/evening attendance\\t"). No se modifica ningún valor.

    Ejemplo:
        datos = utilidades.cargar_estudiantes()
    """
    # Autor: Marcelo Cueto
    ruta = os.path.join(CARPETA_DATOS, "data.csv")
    datos = pd.read_csv(ruta, sep=";", encoding="utf-8-sig")
    datos.columns = datos.columns.str.strip()   # quita la tabulación del nombre
    return datos


def cargar_obesidad():
    """Carga el CSV de obesidad y aplica los pasos determinísticos (diseño, sección 4.1).

    1. Elimina los duplicados exactos del CSV original.
    2. Marca las filas probablemente reales (enteros en la encuesta y en Age).
    3. Redondea a la precisión de la encuesta y fusiona CALC = "Always" con "Frequently".
    4. Crea "grupo": filas idénticas en las 14 variables sin antropometría.

    Ejemplo:
        datos = utilidades.cargar_obesidad()
    """
    # Autor: Marcelo Cueto
    datos = pd.read_csv(os.path.join(CARPETA_DATOS, "ObesityDataSet_raw_and_data_sinthetic.csv"))
    datos = datos.drop_duplicates().reset_index(drop=True)
    columnas_enteras = ENCUESTA_OBESIDAD + ["Age"]
    es_entero = np.isclose(datos[columnas_enteras], datos[columnas_enteras].round())
    datos["probablemente_real"] = es_entero.all(axis=1)
    datos[columnas_enteras] = datos[columnas_enteras].round()
    datos["Height"] = datos["Height"].round(2)
    datos["Weight"] = datos["Weight"].round()
    datos["CALC"] = datos["CALC"].replace("Always", "Frequently")
    sin_antropometria = columnas_por_escenario("obesidad", "E2")
    datos["grupo"] = datos[sin_antropometria].astype(str).agg("|".join, axis=1).factorize()[0]
    return datos


def columnas_por_escenario(dataset, escenario):
    """Devuelve la lista de variables de entrada de cada escenario (diseño, sección 1).

    - estudiantes E1: al matricularse (sin las 12 "Curricular units").
    - estudiantes E2: fin del 2.º semestre (las 36 variables).
    - obesidad E1: con Height y Weight.  obesidad E2: sin Height ni Weight.

    Ejemplo:
        X_train = train[utilidades.columnas_por_escenario("obesidad", "E2")]
    """
    # Autor: Marcelo Cueto
    if dataset == "estudiantes":
        if escenario == "E1":
            return [c for c in COLUMNAS_ESTUDIANTES if not c.startswith("Curricular units")]
        return list(COLUMNAS_ESTUDIANTES)
    if escenario == "E2":
        return [c for c in COLUMNAS_OBESIDAD if c not in ANTROPOMETRIA]
    return list(COLUMNAS_OBESIDAD)


def codificar_objetivo(y, dataset):
    """Convierte las clases del target (texto) en enteros 0, 1, 2...

    Estudiantes: Dropout = 0, Enrolled = 1, Graduate = 2. Obesidad: 0 a 6 en
    orden de severidad (ORDEN_OBESIDAD), así el kappa ponderado usa el orden real.
    Es necesario porque MLPClassifier de scikit-learn 1.6.1 falla con
    early_stopping=True cuando las clases son texto.

    Ejemplo:
        y_train = utilidades.codificar_objetivo(train[utilidades.OBJETIVO_OBESIDAD], "obesidad")
    """
    # Autor: Marcelo Cueto
    clases = CLASES_ESTUDIANTES if dataset == "estudiantes" else ORDEN_OBESIDAD
    return y.map({clase: posicion for posicion, clase in enumerate(clases)})


def particiones_agrupadas(y, grupos, semilla):
    """Devuelve 5 particiones (train, validación) estratificadas y agrupadas (diseño, sección 7).

    Las filas del mismo grupo quedan siempre del mismo lado. StratifiedGroupKFold
    estratifica mejor sin barajar; para obtener particiones distintas con cada
    semilla, antes se renumeran los grupos al azar.

    Ejemplo:
        particiones = utilidades.particiones_agrupadas(y_train, train["grupo"], utilidades.SEMILLA)
    """
    # Autor: Marcelo Cueto
    codigos = pd.factorize(grupos)[0]
    renumerados = np.random.default_rng(semilla).permutation(codigos.max() + 1)[codigos]
    divisor = StratifiedGroupKFold(n_splits=5)
    return list(divisor.split(np.zeros(len(y)), y, renumerados))


def dividir_train_test(datos, dataset):
    """Separa el test de reserva (20%) una sola vez y con la semilla fija (diseño, sección 7).

    Estudiantes: partición estratificada. Obesidad: estratificada y agrupada,
    para que las filas idénticas queden del mismo lado. De las 5 particiones
    agrupadas, el test es la que tiene la distribución de clases más parecida
    a la del total (criterio estructural: no mira ningún modelo).

    Ejemplo:
        train, test = utilidades.dividir_train_test(datos, "obesidad")
    """
    # Autor: Marcelo Cueto
    if dataset == "estudiantes":
        y = datos[OBJETIVO_ESTUDIANTES]
        return train_test_split(datos, test_size=0.2, stratify=y, random_state=SEMILLA)
    y = datos[OBJETIVO_OBESIDAD]
    particiones = particiones_agrupadas(y, datos["grupo"], SEMILLA)
    total = y.value_counts(normalize=True)
    desviaciones = [(y.iloc[test].value_counts(normalize=True) - total).abs().max() for _, test in particiones]
    indices_train, indices_test = particiones[int(np.argmin(desviaciones))]
    return datos.iloc[indices_train], datos.iloc[indices_test]


def crear_preprocesador(dataset, escenario):
    """Crea el ColumnTransformer de codificación y escalado (diseño, sección 4.2).

    Se usa dentro de un Pipeline, así que aprende solo del fold de entrenamiento.

    Ejemplo:
        preprocesador = utilidades.crear_preprocesador("estudiantes", "E1")
    """
    # Autor: Marcelo Cueto
    columnas = columnas_por_escenario(dataset, escenario)
    one_hot = OneHotEncoder(min_frequency=0.01, handle_unknown="infrequent_if_exist", sparse_output=False)
    if dataset == "estudiantes":
        numericas = [c for c in columnas if c not in NOMINALES_ESTUDIANTES + BINARIAS_ESTUDIANTES]
        pasos = [("nominales", one_hot, NOMINALES_ESTUDIANTES),
                 ("binarias", "passthrough", BINARIAS_ESTUDIANTES),
                 ("numericas", StandardScaler(), numericas)]
    else:
        ordinales = Pipeline([("orden", OrdinalEncoder(categories=[ORDEN_CAEC, ORDEN_CALC])),
                              ("escala", StandardScaler())])
        numericas = [c for c in columnas if c not in BINARIAS_OBESIDAD + NOMINALES_OBESIDAD + ["CAEC", "CALC"]]
        pasos = [("binarias", OneHotEncoder(drop="if_binary", sparse_output=False), BINARIAS_OBESIDAD),
                 ("nominales", one_hot, NOMINALES_OBESIDAD),
                 ("ordinales", ordinales, ["CAEC", "CALC"]),
                 ("numericas", StandardScaler(), numericas)]
    return ColumnTransformer(pasos)


def evaluar_cv(modelo, X, y, grupos=None, repeticiones=1, semilla=SEMILLA):
    """Devuelve el macro-F1 de cada fold de una CV estratificada de 5 folds (diseño, secciones 7 y 8).

    - Optuna: repeticiones=1, semilla=SEMILLA (5 valores).
    - Comparativa: repeticiones=3, semilla=SEMILLA_COMPARATIVA (15 valores).
    Si se pasan grupos (obesidad), la CV es además agrupada. Los 5 folds se
    evalúan en paralelo (n_jobs=-1); el resultado es el mismo que en serie.

    Ejemplo:
        puntajes = utilidades.evaluar_cv(pipeline, X_train, y_train, grupos=train["grupo"])
        print(puntajes.mean(), puntajes.std())
    """
    # Autor: Marcelo Cueto
    puntajes = []
    for repeticion in range(repeticiones):
        if grupos is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla + repeticion)
        else:
            cv = particiones_agrupadas(y, grupos, semilla + repeticion)
        puntajes.extend(cross_val_score(modelo, X, y, cv=cv, scoring="f1_macro", n_jobs=-1))
    return np.array(puntajes)


def metricas_test(modelo, X_test, y_test, dataset):
    """Calcula las métricas del diseño (sección 9) para un modelo ya entrenado.

    Incluye las del curso (accuracy, precision, recall), macro-F1, balanced
    accuracy y ROC-AUC OvR. En obesidad agrega el kappa ponderado cuadrático;
    para eso y_test debe venir de codificar_objetivo (códigos en orden de severidad).

    Ejemplo:
        metricas = utilidades.metricas_test(modelo_final, X_test, y_test, "obesidad")
    """
    # Autor: Marcelo Cueto
    y_pred = modelo.predict(X_test)
    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "roc_auc_ovr": roc_auc_score(y_test, modelo.predict_proba(X_test), multi_class="ovr"),
    }
    if dataset == "obesidad":
        metricas["kappa_cuadratico"] = cohen_kappa_score(y_test, y_pred, weights="quadratic")
    return metricas


def registrar_resultado(filas, dataset, escenario, modelo, configuracion, metrica, media, desviacion, particion):
    """Agrega una fila al registro de resultados del notebook (diseño, sección 11.2).

    "filas" es una lista que el notebook crea vacía al inicio; al final se
    guarda con guardar_tabla(pd.DataFrame(filas), "resultados_<dataset>").
    particion: "CV" o "test".

    Ejemplo:
        filas = []
        utilidades.registrar_resultado(filas, "obesidad", "E2", "KNN", "k=15", "f1_macro", 0.61, 0.02, "CV")
    """
    # Autor: Marcelo Cueto
    filas.append({"dataset": dataset, "escenario": escenario, "modelo": modelo,
                  "configuracion": str(configuracion), "metrica": metrica,
                  "media": media, "desviacion": desviacion, "particion": particion})


def guardar_tabla(tabla, nombre):
    """Guarda una tabla como CSV en resultados/tablas/ con su nombre final.

    Ejemplo:
        utilidades.guardar_tabla(tabla_comparativa, "tabla_03_comparativa_cv_obesidad")
    """
    # Autor: Marcelo Cueto
    os.makedirs(CARPETA_TABLAS, exist_ok=True)
    tabla.to_csv(os.path.join(CARPETA_TABLAS, nombre + ".csv"), index=False)


def guardar_figura(figura, nombre):
    """Guarda una figura de matplotlib como PNG (300 dpi) en resultados/figuras/.

    Ejemplo:
        figura, eje = plt.subplots()
        utilidades.guardar_figura(figura, "figura_02_target_obesidad")
    """
    # Autor: Marcelo Cueto
    os.makedirs(CARPETA_FIGURAS, exist_ok=True)
    figura.savefig(os.path.join(CARPETA_FIGURAS, nombre + ".png"), dpi=300, bbox_inches="tight")


def consolidar_resultados():
    """Une resultados_estudiantes.csv y resultados_obesidad.csv en resultados_experimentos.csv.

    La ejecuta Marcelo al final, después de las corridas oficiales.

    Ejemplo:
        utilidades.consolidar_resultados()
    """
    # Autor: Marcelo Cueto
    partes = [pd.read_csv(os.path.join(CARPETA_TABLAS, f"resultados_{d}.csv"))
              for d in ["estudiantes", "obesidad"]]
    consolidado = pd.concat(partes, ignore_index=True)
    guardar_tabla(consolidado, "resultados_experimentos")
    return consolidado
