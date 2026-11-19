"""Carga, estandarización y unificación de los datasets anuales originales (2001-2016).

Este módulo es el punto de entrada de los datos crudos del portal de la Ciudad de Buenos Aires:
    1. Verifica la integridad física de los archivos CSV (descartando respuestas HTML por rate-limit).
    2. Lee cada año tolerando inconsistencias de delimitadores mediante un callback de líneas corruptas.
    3. Normaliza nombres de columnas a mayúsculas y mapea variantes históricas mediante COLUMN_RENAME_MAP.
    4. Garantiza un esquema consistente (TARGET_COLUMNS), rellenando columnas ausentes con NaN.
    5. Detecta y corrige errores geográficos sistemáticos (inversión de LAT y LON).
    6. Concatena los años especificados en un único DataFrame homogéneo y audita la calidad espacial.
"""
import numpy as np
import pandas as pd

from config import (DATA_RAW, RAW_FILE_PATTERN, CSV_PARAMS,
                    COLUMN_RENAME_MAP, TARGET_COLUMNS)
from src.logger_setup import get_logger

logger = get_logger(__name__)


def _warn_bad_line(fields):
    """Callback invocado por pandas al encontrar una línea con cantidad errónea de campos.

    Registra una advertencia con los primeros 80 caracteres de la línea conflictiva
    y devuelve None para instruir al motor de lectura que la descarte.
    """
    # pandas entrega la línea problemática como lista de campos
    logger.warning("Línea descartada por formato inválido: %s", ";".join(fields)[:80])
    return None


def get_raw_path(year: int):
    """Devuelve la ruta absoluta del archivo CSV correspondiente a un año y valida su contenido.

    Comprueba que el archivo exista en `data/raw/` y que no corresponda a una página de error
    HTML (frecuente cuando el portal de datos abiertos bloquea descargas automatizadas).

    Parameters
    ----------
    year : int
        Año del dataset a localizar (ej. 2001, 2016).

    Returns
    -------
    pathlib.Path
        Ruta verificada hacia el archivo CSV crudo.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en el directorio esperado.
    ValueError
        Si el encabezado del archivo contiene etiquetas HTML o mensajes de rechazo.
    """
    path = DATA_RAW / RAW_FILE_PATTERN.format(year=year)
    if not path.exists():
        raise FileNotFoundError(f"[{year}] No se encontró {path}. Descargalo a mano y copialo en data/raw/")

    # Inspección de los primeros 300 bytes para detectar respuestas HTTP/HTML encubiertas
    with open(path, "rb") as f:
        head = f.read(300).lower()
    if b"<html" in head or b"request rejected" in head:
        raise ValueError(f"[{year}] {path.name} contiene HTML y no datos. Descargalo de nuevo a mano.")

    logger.info("[%s] archivo encontrado: %s", year, path.name)
    return path


def standardize_year(year: int) -> pd.DataFrame:
    """Lee el CSV original de un año determinado y alinea su estructura a TARGET_COLUMNS.

    Aplica el motor Python para máxima tolerancia de formato, limpia y pasa a mayúsculas
    los encabezados, renombra según variantes históricas de nombres, completa con NaN
    las variables ausentes en ese período y corrige coordenadas invertidas.

    Parameters
    ----------
    year : int
        Año a procesar y estandarizar.

    Returns
    -------
    pd.DataFrame
        DataFrame con el conjunto exacto de TARGET_COLUMNS y la columna 'AÑO'.
    """
    path = get_raw_path(year)
    # Se usa engine="python" para habilitar on_bad_lines con función personalizada
    df = pd.read_csv(path, engine="python", on_bad_lines=_warn_bad_line, **CSV_PARAMS)
    raw_shape = df.shape

    # Normalización de encabezados: sin espacios residuales y en mayúsculas
    df.columns = df.columns.str.strip().str.upper()
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df["AÑO"] = year

    # Diagnóstico de variables: faltantes obligatorias vs columnas extras descartables
    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        logger.info("[%s] columnas ausentes (se rellenan con NaN): %s", year, missing)
    extra = [c for c in df.columns if c not in TARGET_COLUMNS]
    if extra:
        logger.warning("[%s] columnas no contempladas (se descartan): %s", year, extra)

    # Asegura que todas las TARGET_COLUMNS existan antes de reordenar
    for c in missing:
        df[c] = np.nan
    df = fix_lat_lon(df, year)[TARGET_COLUMNS]

    logger.info("[%s] estandarizado: %s -> %s", year, raw_shape, df.shape)
    return df


def build_unified_dataset(years) -> pd.DataFrame:
    """Construye el dataset consolidado unificando múltiples años en un único DataFrame.

    Estandariza secuencialmente cada año provisto, los concatena verticalmente,
    asegura la consistencia dimensional y ejecuta un chequeo de límites geográficos.

    Parameters
    ----------
    years : iterable of int
        Colección de años a integrar (ej. config.YEARS).

    Returns
    -------
    pd.DataFrame
        DataFrame consolidado con todas las observaciones y columnas homogéneas.
    """
    frames = [standardize_year(y) for y in years]
    df = pd.concat(frames, ignore_index=True)
    assert list(df.columns) == TARGET_COLUMNS, "Columnas inesperadas tras unificar"
    logger.info("Dataset unificado: %s filas, %s columnas", *df.shape)
    
    check_coordinates(df)
    
    return df


def fix_lat_lon(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Corrige filas donde LAT y LON se encuentran intercambiadas.

    En la Ciudad de Buenos Aires, la latitud ronda los -34.6° y la longitud los -58.4°.
    Si |LAT| > |LON| (|latitud| > 50), las columnas vinieron invertidas en el origen
    y se intercambian sus valores vectorialmente con np.where.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'LAT' y 'LON'.
    year : int
        Año en procesamiento para registro en logs.

    Returns
    -------
    pd.DataFrame
        DataFrame con las coordenadas rectificadas.
    """
    lat = pd.to_numeric(df["LAT"], errors="coerce")
    lon = pd.to_numeric(df["LON"], errors="coerce")
    # En CABA: |lat| < 35 y |lon| > 55. Si |lat| > |lon|, están cruzadas.
    swapped = lat.abs() > lon.abs()
    n = int(swapped.sum())
    if n:
        logger.warning("[%s] %d filas con LAT/LON invertidas, se corrigen", year, n)
    df["LAT"] = np.where(swapped, lon, lat)
    df["LON"] = np.where(swapped, lat, lon)
    return df


def log_null_report(df: pd.DataFrame) -> pd.DataFrame:
    """Genera y registra en el log un informe detallado de valores nulos globales y anuales.

    Permite identificar qué columnas no existían o no se recolectaban en determinados
    años (columnas 100% vacías en un año específico).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame unificado a inspeccionar.

    Returns
    -------
    pd.DataFrame
        Matriz de cantidad de nulos agrupada por columna y por año.
    """
    nulls = df.isnull().sum()
    logger.info("Nulos por columna:\n%s", nulls[nulls > 0].to_string())

    por_año = df.drop(columns="AÑO").isnull().groupby(df["AÑO"]).sum()
    # columnas 100% nulas en algún año = esa variable no existe ese año
    for y, fila in por_año.iterrows():
        total = (df["AÑO"] == y).sum()
        vacias = fila[fila == total].index.tolist()
        if vacias:
            logger.warning("[%s] columnas completamente nulas: %s", y, vacias)
    return por_año


def check_coordinates(df: pd.DataFrame) -> None:
    """Verifica si las coordenadas LAT/LON caen dentro del rectángulo envolvente de CABA.

    Rango geográfico de CABA aproximado:
        - Latitud: entre -34.72 y -34.52
        - Longitud: entre -58.54 y -58.33
    Registra en el logger cuántos registros presentan coordenadas fuera de este rango por año.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'LAT', 'LON' y 'AÑO'.
    """
    lat_ok = df["LAT"].between(-34.72, -34.52)
    lon_ok = df["LON"].between(-58.54, -58.33)
    fuera = df["LAT"].notna() & ~(lat_ok & lon_ok)
    por_año = fuera.groupby(df["AÑO"]).sum()
    logger.info("Coordenadas fuera de CABA en total: %d", int(fuera.sum()))
    for y, n in por_año[por_año > 0].items():
        logger.warning("[%s] %d filas con coordenadas fuera de CABA", y, n)
