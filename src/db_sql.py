"""Persistencia relacional en base de datos SQLite (Tabla 1).

Este módulo gestiona el almacenamiento y la lectura de los datos limpios y estandarizados:
    1. save_clean_data(): Exporta el DataFrame procesado a una base SQLite local
       (`data/processed/deptos.db`) en bloques de 10.000 filas (chunksize), reemplazando
       la tabla existente y ejecutando una aserción de cantidad de registros para asegurar
       integridad física contra pérdidas de escritura.
    2. load_clean_data(): Carga la tabla limpia directamente desde SQLite para alimentar
       los pipelines de modelado y notebooks sin re-ejecutar la limpieza desde archivos CSV.
"""
import sqlite3

import pandas as pd
from sqlalchemy import create_engine

from config import SQLITE_PATH, TABLE_CLEAN, DATA_PROCESSED
from src.logger_setup import get_logger

logger = get_logger(__name__)


def save_clean_data(df: pd.DataFrame, table: str = TABLE_CLEAN) -> None:
    """Guarda el DataFrame de datos limpios en una tabla de la base relacional SQLite (Tabla 1).

    Crea el directorio de destino si no existe, escribe los registros en lotes mediante SQLAlchemy
    y realiza una consulta SQL directa con `COUNT(*)` para verificar que todas las filas hayan
    sido persistidas exitosamente.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame resultante del proceso de limpieza (`clean_dataset`).
    table : str, default=config.TABLE_CLEAN
        Nombre de la tabla relacional en la base SQLite.

    Raises
    ------
    ValueError
        Si la cantidad de filas contabilizadas en SQLite no coincide exactamente con el largo de `df`.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{SQLITE_PATH}")
    # Inserción en bloques de 10.000 filas para optimizar el consumo de memoria en SQLite
    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=10_000)

    # Verificación estricta de persistencia mediante conexión nativa sqlite3
    with sqlite3.connect(SQLITE_PATH) as con:
        n = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    logger.info("Tabla '%s' guardada en %s: %d filas", table, SQLITE_PATH.name, n)
    if n != len(df):
        raise ValueError(f"Se esperaban {len(df)} filas y la tabla tiene {n}")


def load_clean_data(table: str = TABLE_CLEAN) -> pd.DataFrame:
    """Lee y carga en memoria la tabla de datos limpios almacenada en la base relacional SQLite.

    Parameters
    ----------
    table : str, default=config.TABLE_CLEAN
        Nombre de la tabla a recuperar de la base SQLite.

    Returns
    -------
    pd.DataFrame
        DataFrame cargado con los datos procesados y listos para modelado.
    """
    engine = create_engine(f"sqlite:///{SQLITE_PATH}")
    df = pd.read_sql_table(table, engine)
    logger.info("Tabla '%s' leída: %s filas, %s columnas", table, *df.shape)
    return df
