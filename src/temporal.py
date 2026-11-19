"""Validación temporal (Out-Of-Time) para evaluar la capacidad de generalización a futuro.

Este módulo implementa el escenario de evaluación temporal:
    - En problemas inmobiliarios y económicos, una partición puramente aleatoria (i.i.d.)
      puede sobreestimar el rendimiento real, ya que el modelo se entrena y evalúa con avisos
      contemporáneos.
    - La partición temporal divide los datos según una cota cronológica (TEMPORAL_CUTOFF):
      se entrena exclusivamente con años históricos (ej. 2001-2013) y se evalúa sobre
      los años futuros no vistos (ej. 2014-2016).
    - Para mantener la comparabilidad metodológica y optimizar recursos, se reutilizan
      los mejores hiperparámetros encontrados en el escenario aleatorio mediante clone().
"""
import time

from sklearn.base import clone

from config import TEMPORAL_CUTOFF
from src.logger_setup import get_logger

logger = get_logger(__name__)


def temporal_split(df, X, y, cutoff: int = TEMPORAL_CUTOFF):
    """Divide las predictoras y el target en una partición estricta por año de publicación.

    Verifica que ningún registro del conjunto de evaluación corresponda a un año igual o anterior
    al conjunto de entrenamiento mediante una aserción estricta de orden temporal.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que contiene la columna 'AÑO'.
    X : pd.DataFrame
        Matriz de características predictoras (alineada por índice con `df`).
    y : pd.Series
        Vector de etiquetas objetivo (alineado por índice con `df`).
    cutoff : int, default=config.TEMPORAL_CUTOFF
        Último año incluido en el conjunto de entrenamiento (ej. 2013).

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
        Tupla con (X_train_temporal, X_test_temporal, y_train_temporal, y_test_temporal).
    """
    es_train = (df["AÑO"] <= cutoff).to_numpy()
    # Verificación de integridad temporal: max(año_train) < min(año_test)
    assert df.loc[es_train, "AÑO"].max() < df.loc[~es_train, "AÑO"].min(), \
        "Violación en el corte temporal: solapamiento de años entre train y test"
        
    X_tr, X_te = X.loc[es_train], X.loc[~es_train]
    y_tr, y_te = y.loc[es_train], y.loc[~es_train]
    
    logger.info("Split temporal: train %d filas (%d-%d) | test %d filas (%d-%d)",
                len(X_tr), df.loc[es_train, "AÑO"].min(), cutoff,
                len(X_te), cutoff + 1, df.loc[~es_train, "AÑO"].max())
    return X_tr, X_te, y_tr, y_te


def fit_temporal(name: str, base_model, X_tr, y_tr, rf_trees: int = 100):
    """Reentrena un modelo con los datos del período histórico preservando sus mejores hiperparámetros.

    Utiliza clone() de scikit-learn para instanciar un pipeline limpio con la misma estructura
    y parámetros óptimos hallados previamente, evitando repetir la búsqueda de hiperparámetros.

    Parameters
    ----------
    name : str
        Identificador del modelo ('logreg', 'random_forest', 'xgboost', 'dummy').
    base_model : Pipeline or estimator
        Modelo de referencia ajustado en el escenario aleatorio.
    X_tr : pd.DataFrame
        Matriz de predictoras históricas (años <= cutoff).
    y_tr : pd.Series
        Etiquetas históricas.
    rf_trees : int, default=100
        Cantidad de árboles para Random Forest en el reentrenamiento temporal.

    Returns
    -------
    estimator
        Nuevo pipeline reentrenado sobre la ventana temporal histórica.
    """
    model = clone(base_model)
    # En Random Forest se asegura paralelismo total y número de estimadores balanceado
    if name == "random_forest":
        model.set_params(clf__n_estimators=rf_trees, clf__n_jobs=-1)
        
    t0 = time.time()
    model.fit(X_tr, y_tr)
    logger.info("[%s] ajuste temporal con %d filas en %.0f s", name, len(X_tr), time.time() - t0)
    return model