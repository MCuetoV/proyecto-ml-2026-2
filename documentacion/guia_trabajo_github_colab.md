# Guía de trabajo del equipo: GitHub, VS Code y Colab

**Proyecto:** Machine Learning (650064), ULima 2026-2
**Repositorio:** https://github.com/MCuetoV/proyecto-ml-2026-2 (privado)
**Actualizado:** 22/11/2026

> Los nombres de los menús de GitHub y Colab están en inglés, tal como aparecen en pantalla.

## 1. Reglas del equipo

1. **Un notebook, un responsable.** Cada integrante edita solo su notebook. Colab sube el archivo completo al guardar y no combina cambios: si dos personas editan el mismo notebook, el último en guardar borra el trabajo del otro.
2. **`utilidades.py` lo edita solo Marcelo**, desde VS Code. Si alguien necesita un cambio, lo pide.
3. **Abrir siempre la última versión desde GitHub**, nunca una pestaña vieja de Colab.
4. **Cada guardado lleva un mensaje claro** de lo que se hizo (por ejemplo, "EDA de obesidad: duplicados y distribución del target"). Ese historial es la evidencia de la sección de Aportes.
5. **Antes de cada entrega:** "Reiniciar y ejecutar todo" y guardar el notebook con las salidas visibles.
6. **Todo el código tiene que ser explicable** por cualquiera del equipo. Antes de cada bloque, una celda de texto con *Qué hacemos* y *Por qué*; después de cada resultado, una celda con *Qué observamos*.
7. **Autoría:** al inicio de cada notebook va una celda con los autores de cada sección, y cada función de `utilidades.py` lleva el comentario `# Autor: ...`.

## 2. Estructura del repositorio

```
proyecto-ml-2026-2/
├── README.md
├── requirements.txt
├── datos/                          ← CSV originales, sin modificar
├── notebooks/
│   ├── 01_estudiantes_exploracion.ipynb
│   ├── 02_estudiantes_modelado.ipynb
│   ├── 03_obesidad_exploracion.ipynb
│   └── 04_obesidad_modelado.ipynb
├── utilidades.py                   ← funciones comunes (preprocesamiento, evaluación, Optuna)
└── resultados/
    ├── tablas/                     ← CSV que alimentan las tablas del reporte
    └── figuras/                    ← PNG que van en el reporte
```

La asignación de responsables a cada notebook queda por definir.

## 3. Primera vez: integrantes que trabajan en Colab

1. **Aceptar la invitación** al repositorio que llega por correo.
2. **Crear un token de acceso** (necesario porque el repositorio es privado):
   - En GitHub: **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**.
   - **Expiration:** 30 días.
   - **Repository access:** *Only select repositories* → `proyecto-ml-2026-2`.
   - **Permissions → Contents:** *Read and write*.
   - Copiar el token. GitHub lo muestra una sola vez.
3. **Guardar el token en Colab:** ícono de la llave (🔑 **Secrets**) en la barra izquierda → **Add new secret** → nombre `GITHUB_TOKEN` → pegar el token → activar **Notebook access**. Nunca escribir el token en el código.

## 4. Rutina diaria en Colab

1. **Abrir:** **File → Open notebook → GitHub** → marcar **Include private repos** → repositorio `MCuetoV/proyecto-ml-2026-2` → tu notebook. La primera vez, autorizar a Colab en GitHub.
2. **Ejecutar la celda 0** (configuración del entorno).
3. Trabajar.
4. **Guardar:** **File → Save a copy in GitHub** → mismo repositorio, rama `main`, **misma ruta del archivo** y mensaje claro → **OK**.

## 5. Primera vez y rutina diaria en VS Code (Marcelo)

**Primera vez:** repositorio clonado en `~/Documents/CURSOS_UNIVERSIDAD/MachineLearning/proyecto-ml-2026-2`, con Git configurado con el correo de la cuenta de GitHub. Después:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

y seleccionar `.venv` como kernel en cada notebook.

**Cada día:**
1. **Antes de trabajar:** `git pull`.
2. **Al terminar:** panel **Source Control** → mensaje claro → **Commit → Sync Changes**.

## 6. Celda 0 de cada notebook

```python
# Celda 0. Configuración del entorno: funciona igual en VS Code y en Colab.
import os
import sys

EN_COLAB = "google.colab" in sys.modules
USUARIO_GITHUB = "MCuetoV"
REPOSITORIO = "proyecto-ml-2026-2"

if EN_COLAB:
    # En Colab: clonar el repositorio (o actualizarlo) con el token guardado en Secrets
    from google.colab import userdata
    token = userdata.get("GITHUB_TOKEN")
    if not os.path.exists(f"/content/{REPOSITORIO}"):
        !git clone -q https://{token}@github.com/{USUARIO_GITHUB}/{REPOSITORIO}.git /content/{REPOSITORIO}
    os.chdir(f"/content/{REPOSITORIO}")
    !git pull -q
    !pip install -q -r requirements.txt
else:
    # En VS Code: si el notebook se abrió desde notebooks/, subir a la raíz del repositorio
    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")

print("Carpeta de trabajo:", os.getcwd())
```

Desde aquí, las rutas son relativas a la raíz y funcionan igual en ambos entornos: `datos/...`, `resultados/...` e `import utilidades`.

**En Colab, la celda 0 trae `utilidades.py` y los datos; el notebook se guarda aparte**, con "Save a copy in GitHub".

## 7. Entrega

Invitar al profesor Edwin Escobedo como colaborador, o volver público el repositorio, antes de enviar el enlace.
