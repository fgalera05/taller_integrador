"""Punto de entrada y orquestador del pipeline: ejecución reproducible mediante Papermill.

Este script automatiza el ciclo completo del trabajo práctico ejecutando los notebooks en orden:
    1. 01_eda.ipynb: Lectura de CSVs crudos, estandarización, limpieza y persistencia en SQLite (Tabla 1).
    2. 02_modelado.ipynb: Split, optimización de hiperparámetros, validación cruzada estratificada,
       evaluación aleatoria y temporal, prueba de leakage y persistencia en MongoDB Atlas (Tablas 2 y 3).

Modos de ejecución (desde la carpeta `taller_integrador/` con `.venv` activado):
    - python main.py --rapido    # Prueba de humo: búsqueda reducida en 5k filas, sin escribir en Atlas.
    - python main.py --reusar    # Reutiliza el archivo serializado `models/results.joblib`.
    - python main.py             # Ejecución exhaustiva completa desde cero.

Papermill inyecta parámetros en tiempo de ejecución y genera una copia sellada con timestamp
en `notebooks/ejecutados/`, preservando evidencia auditable de la corrida.
"""
import argparse
from datetime import datetime

import papermill as pm

from config import BASE_DIR
from src.logger_setup import get_logger, flush_mongo_logs

logger = get_logger("main")
NB_DIR = BASE_DIR / "notebooks"
OUT_DIR = NB_DIR / "ejecutados"   # Carpeta de persistencia para notebooks con outputs generados
KERNEL = "taller_integrador"      # Nombre del kernel de Jupyter registrado en el entorno virtual


def run_notebook(nombre: str, parametros: dict = None, stamp: str = "") -> None:
    """Ejecuta un notebook Jupyter mediante Papermill y almacena una copia sellada con outputs.

    Sustituye los valores de las variables en la celda etiquetada con el tag 'parameters'
    y fija el directorio de trabajo (cwd) en `notebooks/` para mantener consistencia de rutas.

    Parameters
    ----------
    nombre : str
        Nombre del archivo notebook sin la extensión `.ipynb` (ej. '01_eda', '02_modelado').
    parametros : dict, optional
        Diccionario de hiperparámetros y banderas a inyectar en la ejecución.
    stamp : str, default=""
        Marca de tiempo (YYYYMMDD_HHMMSS) para sincronizar los archivos generados en la misma corrida.
    """
    OUT_DIR.mkdir(exist_ok=True)
    salida = OUT_DIR / f"{nombre}_{stamp}.ipynb"
    logger.info("Ejecutando %s.ipynb con parámetros %s", nombre, parametros or {})
    
    # cwd=str(NB_DIR) asegura que 'sys.path.append("..")' dentro de cada notebook resuelva a taller_integrador/
    pm.execute_notebook(
        str(NB_DIR / f"{nombre}.ipynb"),
        str(salida),
        parameters=parametros or {},
        kernel_name=KERNEL,
        cwd=str(NB_DIR),
        log_output=True
    )
    logger.info("Copia ejecutada guardada en %s", salida)


def main(rapido: bool, reusar: bool) -> None:
    """Función principal de orquestación del pipeline end-to-end.

    Configura los parámetros según las banderas de línea de comandos, coordina
    la ejecución secuencial de `01_eda` y `02_modelado`, y asegura el vaciado final
    de logs hacia MongoDB Atlas.

    Parameters
    ----------
    rapido : bool
        Si es True, activa el modo ligero para smoke-testing.
    reusar : bool
        Si es True, reutiliza los modelos entrenados previamente en disco.
    """
    # Marca de tiempo común para emparejar los notebooks generados en esta sesión
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = {}
    
    if rapido and reusar:
        logger.warning("Se pidieron --rapido y --reusar a la vez: se usa solo --rapido")
        
    if rapido:
        # Configuración liviana de prueba rápida: 5.000 filas para tuning y sin modificar MongoDB
        params = dict(
            search_sample_size=5_000,
            rf_n_iter=2,
            xgb_n_iter=2,
            cv_report_sample_size=5_000,
            models_to_run=["logreg", "xgboost"],
            save_to_mongo=False,
            cache_name="results_rapido.joblib"
        )
    elif reusar:
        # Carga los modelos previamente ajustados sin repetir la búsqueda computacional
        params = dict(reuse_results=True)
        
    # Paso 1: Ejecución del pipeline de carga, limpieza y persistencia en SQLite
    logger.info("Iniciando Paso 1: 01_eda.ipynb")
    run_notebook("01_eda", stamp=stamp)
    
    # Paso 2: Ejecución del entrenamiento, evaluación y persistencia en MongoDB
    logger.info("Iniciando Paso 2: 02_modelado.ipynb")
    run_notebook("02_modelado", params, stamp)
    
    # Paso 3: Asegurar la transmisión de logs residuales hacia MongoDB Atlas
    flush_mongo_logs()
    logger.info("Pipeline completado exitosamente para la corrida %s", stamp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Orquestador del pipeline de análisis y modelado de departamentos.")
    ap.add_argument("--rapido", action="store_true", help="Prueba de humo rápida (submuestra y sin escribir en MongoDB)")
    ap.add_argument("--reusar", action="store_true", help="Reutiliza modelos previamente guardados en models/results.joblib")
    a = ap.parse_args()
    main(a.rapido, a.reusar)
