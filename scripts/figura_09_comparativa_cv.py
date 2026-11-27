"""Figura del reporte: macro-F1 de los 15 folds de la CV 5×3 por modelo, escenarios principales.

Lee las tablas 9 que generan los notebooks 02 y 04 (resultados/tablas/tabla_09_puntajes_cv_*.csv).
Uso, desde la raíz del repositorio, después de la corrida oficial:
    python scripts/figura_09_comparativa_cv.py
"""
# Autor: Marcelo Cueto
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.getcwd())
import utilidades  # noqa: E402

PANELES = [("estudiantes", "E1", "Estudiantes, E1 (al matricularse)"),
           ("obesidad", "E2", "Obesidad, E2 (sin estatura ni peso)")]
ETIQUETAS = {"RF (autores)": "Random Forest\n(autores)", "XGBoost (autores)": "XGBoost\n(autores)",
             "Árbol (autores)": "Árbol\n(autores)", "KNN": "KNN", "MLP": "MLP", "XGBoost": "XGBoost"}
COLOR_AUTORES, COLOR_NUEVOS = "#b8c7d9", "#2f5d8a"

figura, ejes = plt.subplots(1, 2, figsize=(10.5, 3.6))
for eje, (dataset, escenario, titulo) in zip(ejes, PANELES):
    tabla = pd.read_csv(f"resultados/tablas/tabla_09_puntajes_cv_{dataset}.csv")
    tabla = tabla[(tabla["escenario"] == escenario) & (tabla["modelo"] != "Dummy")]
    orden = tabla.groupby("modelo")["f1_macro"].mean().sort_values(ascending=False).index.tolist()
    cajas = eje.boxplot([tabla.loc[tabla["modelo"] == m, "f1_macro"] for m in orden], patch_artist=True,
                        tick_labels=[ETIQUETAS[m] for m in orden], showmeans=True, widths=0.55,
                        medianprops={"color": "black"},
                        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5})
    for caja, modelo in zip(cajas["boxes"], orden):
        caja.set_facecolor(COLOR_AUTORES if "autores" in modelo else COLOR_NUEVOS)
        caja.set_alpha(0.85)
    eje.set_title(titulo, fontsize=10)
    eje.tick_params(axis="x", labelsize=8.5)
    eje.grid(axis="y", alpha=0.3)
ejes[0].set_ylabel("Macro-F1 en cada fold (CV 5×3)")
figura.tight_layout()
utilidades.guardar_figura(figura, "figura_09_comparativa_cv")
print("Figura guardada en resultados/figuras/figura_09_comparativa_cv.png")
