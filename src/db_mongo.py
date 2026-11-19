"""Persistencia en MongoDB Atlas (base no relacional).

Colecciones que se escriben (nombres definidos en config.py):
    - parametros          Tabla 3: configuración de cada modelo
    - metricas            Tabla 2: métricas globales, por clase y matriz de confusión
    - predicciones        Tabla 2: una predicción por fila de test y modelo
    - leakage             prueba de fuga de información
    - validacion_cruzada  métricas de cada fold de la validación cruzada
(La colección "logs" la escribe src/mongo_log_handler.py.)

Todos los documentos llevan el run_id de la ejecución y el campo "escenario"
("comun", "aleatorio", "temporal", "leakage" o "validacion_cruzada"). Volver a correr
la misma ejecución reemplaza sus documentos en lugar de duplicarlos.
"""
from datetime import datetime, timezone

import pandas as pd
from pymongo import MongoClient

from config import (MONGO_URI, MONGO_DB, RUN_ID, COL_PREDICTIONS, COL_METRICS, COL_PARAMS,
                    RANDOM_STATE, TEST_SIZE, CV_FOLDS, SCORING, SEARCH_SAMPLE_SIZE,
                    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_NAME, TARGET_BASE,
                    TEMPORAL_CUTOFF, COL_LEAKAGE, COL_CV, CV_REPORT_SAMPLE_SIZE)
from src.logger_setup import get_logger

logger = get_logger(__name__)


def get_db():
    """Conecta con Atlas (usa MONGO_URI del .env) y devuelve la base del proyecto."""
    if not MONGO_URI:
        raise RuntimeError("Falta MONGO_URI en el .env")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")   # falla rápido si las credenciales o la red están mal
    return client[MONGO_DB]


def _now():
    return datetime.now(timezone.utc)


def _replace_run(col, docs, escenario, batch=5000):
    """Reemplaza lo que esta ejecución guardó en `col` para ese escenario.

    Borra los documentos previos con el mismo run_id y escenario (así no se duplican si
    se vuelve a correr), inserta por lotes y verifica que la cantidad guardada coincida.
    """
    filtro = {"run_id": RUN_ID, "escenario": escenario}
    col.delete_many(filtro)
    for i in range(0, len(docs), batch):
        col.insert_many(docs[i:i + batch])
    n = col.count_documents(filtro)
    logger.info("Mongo '%s' [%s]: %d documentos insertados (run_id=%s)", col.name, escenario, n, RUN_ID)
    if n != len(docs):
        raise ValueError(f"'{col.name}': se esperaban {len(docs)} documentos y hay {n}")


def save_params(results: dict, db=None):
    """Tabla 3: configuración y parametrización de cada modelo (un documento por modelo)."""
    db = db or get_db()
    docs = [{
        "run_id": RUN_ID, "escenario": "comun", "timestamp": _now(), "modelo": name,
        "mejores_parametros": r["best_params"],
        "cv_score": r["cv_score"], "scoring": SCORING, "cv_folds": CV_FOLDS,
        "segundos_busqueda": r["search_seconds"], "segundos_ajuste_final": r["fit_seconds"],
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE, "corte_temporal": TEMPORAL_CUTOFF,
        "filas_busqueda": SEARCH_SAMPLE_SIZE,
        "predictoras_numericas": NUMERIC_FEATURES,
        "predictoras_categoricas": CATEGORICAL_FEATURES,
        "target": TARGET_NAME, "target_base": TARGET_BASE,
    } for name, r in results.items()]
    _replace_run(db[COL_PARAMS], docs, "comun")


def save_metrics(all_metrics: list, escenario: str, db=None):
    """Tabla 2 (parte 1): métricas globales, por clase y matriz de confusión de cada modelo."""
    db = db or get_db()
    docs = [{**m, "run_id": RUN_ID, "escenario": escenario, "timestamp": _now()} for m in all_metrics]
    _replace_run(db[COL_METRICS], docs, escenario)


def save_predictions(preds: pd.DataFrame, escenario: str, db=None):
    """Tabla 2 (parte 2): una predicción por fila de test y modelo.

    `id_fila` es la posición de la fila en la tabla datos_limpios de SQLite, lo que
    permite cruzar las dos bases de datos.
    """
    db = db or get_db()
    docs = preds.to_dict("records")
    for d in docs:
        d["run_id"], d["escenario"] = RUN_ID, escenario
    _replace_run(db[COL_PREDICTIONS], docs, escenario)


def save_leakage(df_leak: pd.DataFrame, db=None):
    """Prueba de leakage: una fila por variante de predictoras."""
    db = db or get_db()
    docs = []
    for r in df_leak.to_dict("records"):
        r["accuracy"], r["f1_macro"] = float(r["accuracy"]), float(r["f1_macro"])
        r["delta_vs_modelo_real"] = float(r["delta_vs_modelo_real"])
        # Los hiperparámetros del Random Forest de la prueba están fijos en
        # src/leakage_check.py (RF_PARAMS): si se cambian allá, cambiarlos también acá.
        r.update({"run_id": RUN_ID, "escenario": "leakage", "timestamp": _now(),
                  "modelo": "random_forest", "hiperparametros_rf": {"n_estimators": 100,
                  "min_samples_leaf": 5}, "random_state": RANDOM_STATE})
        docs.append(r)
    _replace_run(db[COL_LEAKAGE], docs, "leakage")


def save_cv(cv_df: pd.DataFrame, db=None):
    """Validación cruzada: una fila por modelo y fold, con las métricas de ese fold.

    `cv_df` es lo que devuelve src.models.cv_report. La media y el desvío por modelo se
    pueden calcular después con una consulta de agregación (ver notebook de demostración).
    """
    db = db or get_db()
    docs = []
    for r in cv_df.to_dict("records"):
        # Las métricas se guardan como float de Python para que MongoDB las acepte
        doc = {k: (float(v) if k not in ("modelo", "fold") else v) for k, v in r.items()}
        doc["fold"] = int(doc["fold"])
        doc.update({"run_id": RUN_ID, "escenario": "validacion_cruzada", "timestamp": _now(),
                    "cv_folds": CV_FOLDS, "filas_muestra": CV_REPORT_SAMPLE_SIZE,
                    "random_state": RANDOM_STATE})
        docs.append(doc)
    _replace_run(db[COL_CV], docs, "validacion_cruzada")


def load_run(collection: str, escenario: str, run_id: str = None, db=None) -> pd.DataFrame:
    """Lee una colección; si no se da run_id, usa la ejecución más reciente."""
    db = db or get_db()
    col = db[collection]
    if run_id is None:
        # Se toma el run_id más grande (los run_id son fecha y hora, así que el último es el más nuevo)
        last = col.find_one({"escenario": escenario}, sort=[("run_id", -1)], projection={"run_id": 1})
        if last is None:
            raise ValueError(f"'{collection}' no tiene documentos con escenario '{escenario}'")
        run_id = last["run_id"]
    df = pd.DataFrame(list(col.find({"run_id": run_id, "escenario": escenario}, {"_id": 0})))
    logger.info("Mongo '%s' [%s]: %d documentos leídos (run_id=%s)", collection, escenario, len(df), run_id)
    return df