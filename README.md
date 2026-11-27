# Proyecto de Machine Learning: deserción estudiantil y niveles de obesidad

Proyecto grupal del curso **Machine Learning (650064)**, Universidad de Lima, ciclo 2026-2.
Docente: Edwin Escobedo Cárdenas. Entrega: viernes 27/11/2026.

Se entrenan y comparan modelos de clasificación para dos conjuntos de datos del repositorio UCI:

| Dataset | Tarea | Escenario principal |
|---|---|---|
| [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697) (Realinho et al., 2022) | Predecir si un estudiante abandona, sigue matriculado o se gradúa (3 clases) | Información disponible **al matricularse** |
| [Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544) (Mendoza Palechor y De la Hoz Manotas, 2019) | Predecir el nivel de obesidad (7 clases ordenadas) | Hábitos y condición física, **sin estatura ni peso** |

Los métodos nuevos frente a los autores de cada dataset son **KNN** y un **perceptrón multicapa (MLP)**, más **XGBoost** en obesidad, ajustados con **Optuna**. Se comparan con el método de los autores (Random Forest y XGBoost en estudiantes; árbol de decisión en obesidad) y con un clasificador trivial. Todas las decisiones metodológicas y su justificación están en [`documentacion/diseno_experimental.md`](documentacion/diseno_experimental.md).

## Integrantes y responsables

| Integrante | Código | Notebook |
|---|---|---|
| Eros Aaron Ricardo Bellido Mera | 20182315 | `03_obesidad_exploracion` |
| Eduardo Gabriel Tello Yparraguirre | 20212654 | `02_estudiantes_modelado` |
| Marcelo David Jesus Cueto Vassallo | 20200614 | `04_obesidad_modelado`, `utilidades.py` y coordinación |
| Edson Gerardo Vasquez Solis | 20193599 | `01_estudiantes_exploracion` |

## Estructura del repositorio

```
proyecto-ml-2026-2/
├── README.md
├── requirements.txt        ← versiones fijadas iguales a las de Colab (Python 3.13)
├── utilidades.py           ← funciones comunes: carga, escenarios, partición, preprocesamiento, evaluación
├── datos/                  ← CSV originales de UCI, sin modificar (licencia CC BY 4.0)
├── notebooks/
│   ├── 01_estudiantes_exploracion.ipynb
│   ├── 02_estudiantes_modelado.ipynb
│   ├── 03_obesidad_exploracion.ipynb
│   └── 04_obesidad_modelado.ipynb
├── resultados/
│   ├── tablas/             ← CSV que alimentan las tablas del reporte
│   └── figuras/            ← PNG que van en el reporte
├── documentacion/          ← consigna, análisis, diseño experimental y guía de trabajo
└── reporte/                ← reporte final (.docx)
```

Cada notebook de exploración describe su dataset y fija la partición train/test; cada notebook de modelado entrena, compara e interpreta los modelos. Los notebooks gemelos tienen las mismas secciones, en el mismo orden que el reporte.

## Cómo ejecutar

### En VS Code (local)

Requiere Python 3.13.

```bash
cd proyecto-ml-2026-2
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt ipykernel
```

En macOS, XGBoost necesita además la librería OpenMP: `brew install libomp`.

Abrir un notebook, elegir `.venv` como kernel y ejecutar todas las celdas. La celda 0 ubica la carpeta de trabajo en la raíz del repositorio.

### En Google Colab

1. Guardar en *Secrets* (ícono de la llave): `GITHUB_TOKEN`, `GITHUB_NOMBRE` y `GITHUB_CORREO`, con *Notebook access* activado.
2. *File → Open notebook → GitHub*, marcar *Include private repos* y abrir el notebook desde `MCuetoV/proyecto-ml-2026-2`.
3. Ejecutar todo. La celda 0 clona el repositorio, instala `requirements.txt` y trae `utilidades.py` y los datos.

El detalle del flujo de trabajo está en [`documentacion/guia_trabajo_github_colab.md`](documentacion/guia_trabajo_github_colab.md).

## Reproducibilidad

- Semilla única: `utilidades.SEMILLA = 42`.
- Partición train/test única (20% de test), creada por `utilidades.dividir_train_test`.
- Versiones de las librerías fijadas en `requirements.txt`, iguales en Colab y en VS Code.

## Fuentes de los datos

- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data, 7*(11), 146. https://doi.org/10.3390/data7110146
- Mendoza Palechor, F., & De la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. *Data in Brief, 25*, 104344. https://doi.org/10.1016/j.dib.2019.104344
