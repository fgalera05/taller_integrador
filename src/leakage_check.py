"""Evaluación empírica y diagnóstico de fuga de información (Data Leakage).

Este módulo demuestra experimentalmente el impacto de contaminar las predictoras:
    - La variable objetivo (SEGMENTO_PRECIO) se deriva del PRECIO_M2 dentro de cada año.
    - Si un modelo accede a DOLARES, AÑO o directamente a PRECIO_M2, memoriza la regla
      de derivación del target en lugar de aprender los patrones intrínsecos del inmueble
      (superficie, ambientes, comodidades, barrio).
    - Para evidenciar este fenómeno sin sesgos por algoritmo o partición, se entrena exactamente
      el mismo Random Forest sobre el mismo split 80/20, variando únicamente las predictoras disponibles.
    - Se computa la brecha (delta) contra el 'modelo real' para cuantificar la inflación espuria de métricas.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_NAME, RANDOM_STATE
from src.features import build_preprocessor
from src.logger_setup import get_logger

logger = get_logger(__name__)

# Definición de las variantes a comparar contra el modelo limpio de producción
VARIANTES = {
    "sin DOLARES (modelo real)": [],
    "sin DOLARES, con AÑO": ["AÑO"],
    "con DOLARES": ["DOLARES"],
    "con DOLARES + AÑO": ["DOLARES", "AÑO"],
    "con PRECIO_M2 + AÑO": ["PRECIO_M2", "AÑO"],
}

# Parámetros fijos para asegurar comparabilidad estricta entre variantes
RF_PARAMS = dict(n_estimators=100, min_samples_leaf=5)


def leakage_check(df: pd.DataFrame, sample: int = 40_000) -> pd.DataFrame:
    """Ejecuta el experimento sistemático de fuga de información y mide el impacto en el rendimiento.

    Toma una submuestra representativa para optimizar el tiempo de cómputo, realiza una partición
    estratificada 80/20 común y evalúa secuencialmente las cinco variantes de predictoras
    utilizando la misma arquitectura de preprocesamiento y clasificador Random Forest.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset unificado y limpio con todas las columnas originales y creadas.
    sample : int, default=40_000
        Número de filas muestreadas para la prueba.

    Returns
    -------
    pd.DataFrame
        Tabla comparativa con columnas:
            - 'variante': Nombre descriptivo de la prueba.
            - 'predictoras_extra': Columnas agregadas a NUMERIC_FEATURES.
            - 'accuracy': Exactitud en el conjunto de test.
            - 'f1_macro': F1-Score ponderado macro en test.
            - 'filas_train' y 'filas_test': Dimensiones de la partición.
            - 'delta_vs_modelo_real': Diferencia de accuracy respecto a la versión sin variables contaminadas.
    """
    # Muestreo estratificado reproducible sobre el dataset
    d = df.sample(min(sample, len(df)), random_state=RANDOM_STATE)
    y = d[TARGET_NAME].astype(str)
    
    # Se incluyen temporalmente las variables prohibidas para poder inyectarlas selectivamente
    cols = NUMERIC_FEATURES + ["DOLARES", "AÑO", "PRECIO_M2"] + CATEGORICAL_FEATURES
    X = d[cols].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype(object)
        
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=RANDOM_STATE)
    logger.info("Leakage check: muestra de %d filas (train %d, test %d)",
                len(d), len(X_tr), len(X_te))

    rows = []
    for label, extra in VARIANTES.items():
        # Preprocesador dinámico que acepta las predictoras adicionales
        num = NUMERIC_FEATURES + extra
        feats = num + CATEGORICAL_FEATURES
        pipe = Pipeline([
            ("prep", build_preprocessor(num, CATEGORICAL_FEATURES)),
            ("clf", RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE, **RF_PARAMS))
        ])
        pipe.fit(X_tr[feats], y_tr)
        pred = pipe.predict(X_te[feats])
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro")
        
        rows.append({
            "variante": label,
            "predictoras_extra": extra,
            "accuracy": acc,
            "f1_macro": f1,
            "filas_train": len(X_tr),
            "filas_test": len(X_te)
        })
        logger.info("Leakage check [%s]: accuracy=%.3f f1_macro=%.3f", label, acc, f1)

    out = pd.DataFrame(rows)
    # Cálculo del delta de precisión contra el modelo real (sin fuga)
    base = out.loc[out["variante"] == "sin DOLARES (modelo real)", "accuracy"].iloc[0]
    out["delta_vs_modelo_real"] = out["accuracy"] - base
    return out