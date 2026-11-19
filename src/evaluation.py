"""Evaluación exhaustiva de modelos de clasificación multiclase.

Este módulo computa el conjunto de métricas para la Tabla 2 y el dashboard interactivo:
    1. Métricas globales: Exactitud (Accuracy), Precisión, Recall y F1-Score con ponderación
       macro (trata a todas las clases por igual) y weighted (pondera por soporte).
    2. Métricas desagregadas por clase: Precisión, Recall, F1 y soporte para 'bajo', 'medio' y 'alto'.
    3. Matriz de confusión: Con orden explícito de filas (valores reales) y columnas (predicciones).
    4. Diagnóstico de sobreajuste (Overfitting): Evalúa el modelo sobre una submuestra estratificada
       del conjunto de entrenamiento para medir la brecha (gap) con respecto al conjunto de prueba.
    5. Distribución de probabilidades: Extrae el vector de predict_proba para cada observación
       evaluada, permitiendo análisis fino de incertidumbre y curvas de calibración.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

from config import SEGMENT_LABELS, RANDOM_STATE
from src.logger_setup import get_logger

logger = get_logger(__name__)


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test, overfit_sample: int = 20_000):
    """Evalúa un clasificador sobre el conjunto de test y diagnostica sobreajuste en train.

    Calcula métricas globales y desagregadas por clase, construye la matriz de confusión,
    compara la exactitud de test frente a una submuestra de train y extrae las probabilidades
    asignadas a cada segmento.

    Parameters
    ----------
    name : str
        Identificador del modelo (ej. 'logreg', 'random_forest', 'xgboost', 'dummy').
    model : estimator object
        Pipeline o modelo entrenado que implementa predict() y predict_proba().
    X_train : pd.DataFrame
        Matriz de características de entrenamiento (para muestreo de sobreajuste).
    y_train : pd.Series
        Etiquetas reales de entrenamiento.
    X_test : pd.DataFrame
        Matriz de características de evaluación.
    y_test : pd.Series
        Etiquetas reales de evaluación.
    overfit_sample : int, default=20_000
        Tamaño máximo de la submuestra estratificada de train para control rápido de sobreajuste.

    Returns
    -------
    tuple of (dict, pd.DataFrame)
        - metrics (dict): Diccionario con métricas agregadas, por clase y matriz de confusión
          serializable en JSON (para guardar en la colección 'metricas' de MongoDB).
        - preds (pd.DataFrame): DataFrame con cada predicción individual, probabilidades por clase
          y columna booleana 'acierto' (para la colección 'predicciones').
    """
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test), "Predicciones y test con distinto largo"

    # 1. Métricas globales en test
    acc = accuracy_score(y_test, y_pred)
    # macro: promedio simple entre clases (prioriza que el modelo prediga bien todas)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=SEGMENT_LABELS, average="macro", zero_division=0)
    # weighted: pondera por la frecuencia de cada clase en el conjunto de prueba
    pw, rw, fw, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=SEGMENT_LABELS, average="weighted", zero_division=0)
    # métricas individuales por clase
    pc, rc, fc, sc = precision_recall_fscore_support(
        y_test, y_pred, labels=SEGMENT_LABELS, zero_division=0)

    # 2. Control de sobreajuste: evaluar train_sample sin costo computacional excesivo
    X_s, _, y_s, _ = train_test_split(X_train, y_train, train_size=min(overfit_sample, len(X_train)),
                                      stratify=y_train, random_state=RANDOM_STATE)
    acc_train = accuracy_score(y_s, model.predict(X_s))
    gap = acc_train - acc
    if gap > 0.05:
        # Alerta si el rendimiento en entrenamiento supera al de test por más de 5 puntos porcentuales
        logger.warning("[%s] posible sobreajuste: accuracy train %.3f vs test %.3f", name, acc_train, acc)

    # 3. Matriz de confusión estructurada
    cm = confusion_matrix(y_test, y_pred, labels=SEGMENT_LABELS)
    metrics = {
        "model": name,
        "accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f,
        "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
        "accuracy_train_sample": acc_train,
        "per_class": {l: {"precision": float(pc[i]), "recall": float(rc[i]),
                          "f1": float(fc[i]), "support": int(sc[i])}
                      for i, l in enumerate(SEGMENT_LABELS)},
        "confusion_matrix": cm.tolist(),
        "labels": SEGMENT_LABELS,
    }
    logger.info("[%s] TEST accuracy=%.4f precision=%.4f recall=%.4f f1_macro=%.4f (train acc %.4f)",
                name, acc, p, r, f, acc_train)
    logger.info("[%s] matriz de confusión (filas=real, columnas=predicho %s):\n%s",
                name, SEGMENT_LABELS, cm)

    # 4. Probabilidades individuales por clase y registro de acierto
    proba = model.predict_proba(X_test)
    preds = pd.DataFrame({"id_fila": X_test.index, "modelo": name,
                          "y_real": y_test.values, "y_pred": y_pred})
    for i, c in enumerate(model.classes_):
        preds[f"proba_{c}"] = proba[:, i].round(4)
    preds["acierto"] = preds["y_real"] == preds["y_pred"]
    return metrics, preds