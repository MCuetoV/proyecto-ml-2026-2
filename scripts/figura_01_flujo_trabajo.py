"""Figura 1 del reporte: flujo de trabajo seguido en ambos conjuntos de datos.

No usa datos: dibuja los pasos de la Metodología (diseño experimental, §11.2).
Uso, desde la raíz del repositorio:
    python scripts/figura_01_flujo_trabajo.py
"""
# Autor: Marcelo Cueto
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.getcwd())
import utilidades  # noqa: E402

PASOS = [
    ("1. Exploración y calidad", "Faltantes, duplicados,\nclases, huella sintética"),
    ("2. Escenarios", "Según la información\ndisponible en cada momento"),
    ("3. Partición", "Test reservado de 20 %\n(agrupado en obesidad)"),
    ("4. Preprocesamiento", "Pipeline: codificación,\nescalado, SMOTENC, L1"),
    ("5. Modelado", "Dummy y autores (Grid)\nKNN, MLP y XGBoost (Optuna)"),
    ("6. Comparativa", "CV 5×3, Wilcoxon,\nablaciones y sensibilidad"),
    ("7. Test e interpretación", "Métricas finales, confusión,\npermutación, aprendizaje"),
]
ANCHO, ALTO = 3.2, 1.45
POSICIONES = [(0.15 + 3.55 * i, 2.3) for i in range(4)] + [(1.925 + 3.55 * i, 0.2) for i in range(3)]


def dibujar_caja(eje, x, y, titulo, detalle):
    # Autor: Marcelo Cueto
    eje.add_patch(FancyBboxPatch((x, y), ANCHO, ALTO, boxstyle="round,pad=0.03,rounding_size=0.12",
                                 facecolor="#e8f0f8", edgecolor="#2f5d8a", linewidth=1.2))
    eje.text(x + ANCHO / 2, y + ALTO - 0.32, titulo, ha="center", va="center", fontsize=9, fontweight="bold")
    eje.text(x + ANCHO / 2, y + 0.52, detalle, ha="center", va="center", fontsize=8.2, linespacing=1.3)


def flecha(eje, inicio, fin):
    # Autor: Marcelo Cueto
    eje.add_patch(FancyArrowPatch(inicio, fin, arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#2f5d8a"))


figura, eje = plt.subplots(figsize=(10.5, 3.9))
for (titulo, detalle), (x, y) in zip(PASOS, POSICIONES):
    dibujar_caja(eje, x, y, titulo, detalle)
for i in [0, 1, 2, 4, 5]:
    (x1, y1), (x2, y2) = POSICIONES[i], POSICIONES[i + 1]
    flecha(eje, (x1 + ANCHO, y1 + ALTO / 2), (x2, y2 + ALTO / 2))
(x4, y4), (x5, y5) = POSICIONES[3], POSICIONES[4]
medio = (y4 + y5 + ALTO) / 2   # paso de la fila 1 a la fila 2 por el espacio libre entre ambas
eje.plot([x4 + ANCHO / 2, x4 + ANCHO / 2, x5 + ANCHO / 2], [y4, medio, medio], color="#2f5d8a", linewidth=1.2)
flecha(eje, (x5 + ANCHO / 2, medio), (x5 + ANCHO / 2, y5 + ALTO))
eje.set_xlim(0, 14.4)
eje.set_ylim(0, 3.95)
eje.axis("off")
utilidades.guardar_figura(figura, "figura_01_flujo_trabajo")
print("Figura guardada en resultados/figuras/figura_01_flujo_trabajo.png")
