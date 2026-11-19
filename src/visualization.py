"""Generación y exportación de gráficos para el informe técnico y análisis de resultados.

Este módulo produce el catálogo visual persistido en `reports/figures/`:
    1. plot_price_by_year(): Muestra la mediana y terciles de PRECIO_M2 a lo largo de 2001-2016,
       justificando empíricamente por qué la variable objetivo debe calcularse por año.
    2. plot_confusion_matrices(): Grilla de mapas de calor con porcentajes normalizados por fila
       y conteos absolutos para comparar errores entre clases en cada escenario.
    3. plot_model_comparison(): Gráfico de barras facetado por escenario (aleatorio vs temporal)
       para contrastar Accuracy, Precisión, Recall y F1-Score entre clasificadores.
    4. plot_accuracy_by_year(): Analiza la estabilidad del rendimiento año a año en el escenario
       temporal para detectar degradación por desfase temporal o shocks macroeconómicos.
    5. plot_permutation_importance(): Computa y grafica la caída de exactitud al permutar cada variable,
       ofreciendo interpretabilidad agnóstica al modelo sin sesgo de cardinalidad.
    6. plot_leakage(): Compara visualmente el salto artificial de rendimiento entre el modelo limpio
       y las variantes contaminadas con variables de precio o temporales.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from config import FIGURES_DIR, SEGMENT_LABELS, RANDOM_STATE
from src.logger_setup import get_logger

logger = get_logger(__name__)
# Estilo base homogéneo para todas las figuras del informe
sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, name: str) -> None:
    """Guarda la figura en el directorio configurado con resolución apta para informes (150 DPI).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Objeto figura a exportar.
    name : str
        Nombre del archivo sin extensión.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Figura guardada: %s", path.name)


def plot_price_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Grafica la evolución histórica de los terciles y la mediana de PRECIO_M2 en CABA.

    Permite observar la tendencia inflacionaria en dólares del mercado inmobiliario
    y justifica por qué definir los segmentos ('bajo', 'medio', 'alto') de forma anual
    previene que los años recientes queden desbalanceados hacia la clase 'alta'.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio que contiene columnas 'AÑO' y 'PRECIO_M2'.

    Returns
    -------
    pd.DataFrame
        Tabla con los valores de corte (tercil 1/3, mediana y tercil 2/3) calculados por año.
    """
    q = df.groupby("AÑO")["PRECIO_M2"].quantile([1/3, 0.5, 2/3]).unstack()
    q.columns = ["tercil 1/3", "mediana", "tercil 2/3"]
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    q.plot(ax=ax, marker="o")
    ax.set(title="PRECIO_M2 (USD/m²) por año y cortes del target", ylabel="USD/m²")
    _save(fig, "precio_m2_por_anio")
    return q


def plot_confusion_matrices(metrics: pd.DataFrame, escenario: str) -> None:
    """Genera una hilera de matrices de confusión normalizadas por fila para los modelos de un escenario.

    Muestra tanto el conteo absoluto de avisos como el porcentaje de acierto/error sobre
    el total real de cada fila (clase verdadera).

    Parameters
    ----------
    metrics : pd.DataFrame
        DataFrame de métricas que incluye la columna 'confusion_matrix' y 'model'.
    escenario : str
        Nombre del escenario ('aleatorio' o 'temporal') para el título y nombre de archivo.
    """
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2))
    axes = np.atleast_1d(axes)
    
    for ax, (_, r) in zip(axes, metrics.iterrows()):
        cm = np.array(r["confusion_matrix"])
        # Normalización por fila (recall por clase)
        pct = cm / cm.sum(axis=1, keepdims=True)
        # Anotación combinada: conteo absoluto + porcentaje relativo
        annot = np.array([[f"{cm[i, j]}\n{pct[i, j]:.0%}" for j in range(3)] for i in range(3)])
        sns.heatmap(pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    xticklabels=SEGMENT_LABELS, yticklabels=SEGMENT_LABELS, ax=ax)
        ax.set(title=f"{r['model']} (acc {r['accuracy']:.3f})", xlabel="Predicho", ylabel="Real")
        
    fig.suptitle(f"Matrices de confusión - escenario {escenario} (porcentaje por fila)")
    _save(fig, f"matrices_confusion_{escenario}")


def plot_model_comparison(metrics_all: pd.DataFrame) -> None:
    """Construye un gráfico de barras comparativo facetado por escenario de evaluación.

    Compara lado a lado exactitud, precisión macro, recall macro y F1 macro entre
    los modelos participantes.

    Parameters
    ----------
    metrics_all : pd.DataFrame
        DataFrame consolidado de métricas que contiene registros de ambos escenarios.
    """
    cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    long = metrics_all.melt(id_vars=["escenario", "model"], value_vars=cols,
                            var_name="métrica", value_name="valor")
    g = sns.catplot(data=long, x="métrica", y="valor", hue="model", col="escenario",
                    kind="bar", height=4, aspect=1.4)
    g.set(ylim=(0, 1))
    g.figure.suptitle("Comparación de modelos", y=1.03)
    _save(g.figure, "comparacion_modelos")


def plot_accuracy_by_year(preds: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Evalúa y grafica la tasa de acierto (accuracy) desagregada por año en el split temporal.

    Permite identificar si los modelos entrenados con datos históricos sufren degradación
    gradual a medida que se alejan del año de corte (2013).

    Parameters
    ----------
    preds : pd.DataFrame
        Predicciones del escenario temporal generadas por evaluate_model().
    df : pd.DataFrame
        Dataset limpio para asociar cada fila con su respectivo año de publicación.

    Returns
    -------
    pd.DataFrame
        Tabla con la exactitud promedio de cada modelo por año del período de prueba.
    """
    p = preds.copy()
    p["AÑO"] = df.loc[p["id_fila"], "AÑO"].to_numpy()
    t = p.groupby(["AÑO", "modelo"])["acierto"].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(7, 4))
    t.plot(kind="bar", ax=ax)
    ax.set(title="Accuracy por año (escenario temporal)", ylabel="accuracy", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=0)
    _save(fig, "accuracy_por_anio_temporal")
    return t.round(3)


def plot_permutation_importance(model, X_test: pd.DataFrame, y_test: pd.Series, n: int = 8000) -> pd.DataFrame:
    """Calcula y grafica la importancia de características mediante permutación de variables.

    A diferencia de la importancia basada en impureza (MDI), la permutación evalúa directamente
    la caída de precisión en el conjunto de test al aleatorizar cada columna, evitando
    sobrestimar variables de alta cardinalidad o continuas.

    Parameters
    ----------
    model : estimator object
        Modelo o pipeline ajustado.
    X_test : pd.DataFrame
        Matriz de características de test.
    y_test : pd.Series
        Etiquetas de test.
    n : int, default=8000
        Tamaño de la muestra de evaluación para agilizar el cálculo de permutaciones.

    Returns
    -------
    pd.DataFrame
        DataFrame ordenado descendentemente por caída promedio de accuracy y su desvío estándar.
    """
    idx = X_test.sample(min(n, len(X_test)), random_state=RANDOM_STATE).index
    res = permutation_importance(model, X_test.loc[idx], y_test.loc[idx], scoring="accuracy",
                                 n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
    imp = (pd.DataFrame({
        "variable": X_test.columns,
        "caida_accuracy": res.importances_mean,
        "desvio": res.importances_std
    }).sort_values("caida_accuracy", ascending=True))
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp["variable"], imp["caida_accuracy"], xerr=imp["desvio"])
    ax.set(title="Importancia por permutación (Random Forest)", xlabel="caída de accuracy")
    _save(fig, "importancia_variables")
    return imp.iloc[::-1].reset_index(drop=True)


def plot_leakage(leak: pd.DataFrame) -> None:
    """Genera un gráfico de barras horizontales contrastando las variantes del leakage check.

    Incluye una línea de referencia en 0.333 (rendimiento del azar en 3 clases) para ilustrar
    visualmente el incremento desproporcionado de accuracy cuando se introducen predictoras prohibidas.

    Parameters
    ----------
    leak : pd.DataFrame
        Salida generada por la función `leakage_check()`.
    """
    d = leak.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(d["variante"], d["accuracy"])
    ax.axvline(1 / 3, color="gray", ls="--", label="azar (0,333)")
    ax.set(title="Prueba de leakage: accuracy según predictoras", xlabel="accuracy", xlim=(0, 1))
    ax.legend()
    _save(fig, "leakage")
