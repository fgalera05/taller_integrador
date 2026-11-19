"""Ingeniería de características, partición de datos y ensamblado del preprocesador.

Este módulo encapsula la preparación de matrices de diseño para el modelado:
    1. get_xy(): Separa la matriz de predictoras (X) y el vector objetivo (y),
       verificando que ninguna variable prohibida (NON_FEATURES: DOLARES, PRECIO_M2, AÑO, etc.)
       se filtre en el conjunto predictivo para evitar data leakage por diseño.
    2. split_data(): Genera la partición estratificada en entrenamiento y evaluación (80/20).
    3. build_preprocessor(): Construye un ColumnTransformer de scikit-learn con ramas diferenciadas
       para variables numéricas y categóricas. La imputación y el escalado se integran en este
       Pipeline para que cualquier transformación aprenda exclusivamente de los datos de entrenamiento.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (NUMERIC_FEATURES, CATEGORICAL_FEATURES, NON_FEATURES,
                    TARGET_NAME, TEST_SIZE, RANDOM_STATE)
from src.logger_setup import get_logger

logger = get_logger(__name__)


def get_xy(df):
    """Separa el DataFrame en la matriz de predictoras (X) y el vector objetivo (y).

    Aplica una validación de seguridad contra Data Leakage: si alguna de las columnas
    en `config.NON_FEATURES` (como 'DOLARES', 'PRECIO_M2' o identificadores) está presente
    en las predictoras seleccionadas, lanza una excepción inmediata para abortar el entrenamiento.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame limpio que contiene tanto predictoras como la variable objetivo.

    Returns
    -------
    tuple of (pd.DataFrame, pd.Series)
        - X: Matriz con las variables numéricas y categóricas seleccionadas.
        - y: Serie con la etiqueta de clase ('bajo', 'medio', 'alto') en formato string.

    Raises
    ------
    ValueError
        Si se detectan columnas prohibidas en la lista de características.
    """
    feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    prohibidas = set(feats) & set(NON_FEATURES)
    if prohibidas:
        raise ValueError(f"Predictoras prohibidas (leakage/identificadores): {prohibidas}")

    X = df[feats].copy()
    for c in CATEGORICAL_FEATURES:
        # Cast explícito a object para evitar conflictos con el tipo StringDtype de pandas
        X[c] = X[c].astype(object)
    y = df[TARGET_NAME].astype(str)
    logger.info("X: %s | y: %s | predictoras: %s", X.shape, y.shape, feats)
    return X, y


def split_data(X, y):
    """Realiza la partición estratificada entre conjuntos de entrenamiento y prueba.

    Garantiza que la distribución relativa de las clases del segmento de valor
    ('bajo', 'medio', 'alto') sea idéntica tanto en train como en test mediante `stratify=y`.

    Parameters
    ----------
    X : pd.DataFrame
        Matriz de características predictoras.
    y : pd.Series
        Vector de etiquetas objetivo.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
        Tupla con (X_train, X_test, y_train, y_test).
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    logger.info("Train: %d filas | Test: %d filas", len(X_tr), len(X_te))
    logger.info("Target en train:\n%s", y_tr.value_counts(normalize=True).round(3).to_string())
    logger.info("Target en test:\n%s", y_te.value_counts(normalize=True).round(3).to_string())
    return X_tr, X_te, y_tr, y_te


def build_preprocessor(numeric_features=None, categorical_features=None) -> ColumnTransformer:
    """Construye el ColumnTransformer de scikit-learn para preprocesamiento heterogéneo.

    Estrategia de transformación:
        - Variables numéricas:
          1. SimpleImputer(strategy='median', add_indicator=True): Imputa con la mediana
             (robusta a outliers) y añade columnas binarias auxiliares que registran
             dónde existía un dato faltante, permitiendo a los modelos capturar patrones de ausencia.
          2. StandardScaler(): Centra con media 0 y escala a varianza unitaria (esencial
             para convergencia de Regresión Logística y cálculo de distancias).
        - Variables categóricas:
          1. SimpleImputer(strategy='constant', fill_value='SIN_DATO'): Trata los valores
             faltantes como una categoría explícita.
          2. OneHotEncoder(handle_unknown='ignore', min_frequency=20): Codificación en dummies,
             agrupando categorías poco frecuentes (<20 avisos) y descartando categorías no vistas
             en inferencia sin generar error.

    Parameters
    ----------
    numeric_features : list of str, optional
        Lista personalizada de variables numéricas (por defecto config.NUMERIC_FEATURES).
    categorical_features : list of str, optional
        Lista personalizada de variables categóricas (por defecto config.CATEGORICAL_FEATURES).

    Returns
    -------
    ColumnTransformer
        Preprocesador configurado listo para integrarse en un Pipeline de scikit-learn.
    """
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="SIN_DATO")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    
    return ColumnTransformer([
        ("num", numeric, numeric_features),
        ("cat", categorical, categorical_features),
    ])