"""Limpieza del dataset unificado y creación de la variable objetivo.

clean_dataset() aplica estos pasos, en este orden:
    1. drop_unused_columns       descarta columnas casi vacías
    2. drop_duplicates           elimina avisos duplicados exactos
    3. clean_prices_and_surface  convierte a número, filtra outliers y calcula PRECIO_M2
    4. clean_text_columns        unifica BARRIO y ORIENT (mayúsculas y variantes de escritura)
    5. clean_coordinates         pasa a NaN las coordenadas fuera de CABA
    6. clean_amenities           BAULERA, COCHERA, LAVADERO y TERRAZA pasan a 0/1
    7. clean_antiguedad          "A ESTRENAR" = 0; el 0 numérico se trata como dato faltante
    8. clean_counts              AMBIENTES y BAÑOS inválidos pasan a NaN
    9. create_target             terciles de PRECIO_M2 calculados dentro de cada año

Los faltantes (NaN) NO se imputan acá: se imputan dentro del Pipeline de scikit-learn
(src/features.py), para que la imputación se aprenda solo con los datos de entrenamiento.
Cada paso deja en el log cuántas filas o valores modificó.
"""
import numpy as np
import pandas as pd

from config import (DROP_COLUMNS, BINARY_COLUMNS, CABA_LAT, CABA_LON, M2_RANGE,
                    DOLARES_RANGE, USM2_RANGE, MAX_AMBIENTES, MAX_BANOS,
                    MAX_ANTIGUEDAD, TARGET_BASE, TARGET_NAME, SEGMENT_LABELS,
                    BARRIO_FIXES, ORIENT_FIXES, ORIENT_INVALID)
from src.logger_setup import get_logger

logger = get_logger(__name__)


def _log_rows(step, before, after):
    """Registra en el log cuántas filas eliminó un paso de la limpieza."""
    logger.info("%s: %d -> %d filas (%d eliminadas)", step, before, after, before - after)


def drop_unused_columns(df):
    """Descarta las columnas con más del 90% de nulos (lista en config.DROP_COLUMNS)."""
    cols = [c for c in DROP_COLUMNS if c in df.columns]
    logger.info("Se descartan columnas casi vacías: %s", cols)
    return df.drop(columns=cols)


def drop_duplicates(df):
    """Elimina avisos idénticos en todas las columnas (publicaciones repetidas)."""
    n0 = len(df)
    df = df.drop_duplicates()
    _log_rows("Duplicados exactos", n0, len(df))
    return df


def clean_prices_and_surface(df):
    """Convierte M2, DOLARES y U_S_M2 a número, elimina outliers y crea PRECIO_M2.

    PRECIO_M2 = DOLARES / M2 es la base del target. Se recalcula porque el U_S_M2
    publicado no es consistente entre años (en 2016 usa los m² cubiertos).
    """
    # Texto -> número. 2015 usa coma decimal (por ejemplo '1666,7'), por eso se cambia por punto;
    # lo que no se puede convertir queda NaN.
    for c in ["M2", "DOLARES", "U_S_M2"]:
        s = df[c].astype("string").str.strip().str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(s, errors="coerce").astype("float64")

    # Sin superficie o precio no se puede calcular el target: se eliminan esas filas
    n0 = len(df)
    df = df.dropna(subset=["M2", "DOLARES"])
    _log_rows("Sin M2 o precio", n0, len(df))

    # rangos de superficie y precio total
    ok = df["M2"].between(*M2_RANGE) & df["DOLARES"].between(*DOLARES_RANGE)
    n0 = len(df)
    df = df[ok].copy()
    _log_rows("Outliers de M2/DOLARES", n0, len(df))

    # precio por m² propio, consistente en todos los años
    # (en 2016 el U_S_M2 publicado está calculado sobre m² cubiertos y no es comparable)
    df["PRECIO_M2"] = df["DOLARES"] / df["M2"]
    ok = df["PRECIO_M2"].between(*USM2_RANGE)
    # se avisa por año cuántas filas quedan fuera de rango antes de eliminarlas
    por_año = (~ok).groupby(df["AÑO"]).sum()
    for y, n in por_año[por_año > 0].items():
        logger.warning("[%s] %d filas con PRECIO_M2 fuera de rango", y, n)
    n0 = len(df)
    df = df[ok]
    _log_rows("Outliers de PRECIO_M2", n0, len(df))

    # solo informativo: U_S_M2 publicado vs. recalculado
    desvio = ((df["PRECIO_M2"] - df["U_S_M2"]).abs() / df["U_S_M2"]) > 0.05
    # se avisa solo si hay más de 100 filas con diferencia (en la práctica, 2016)
    for y, n in desvio.groupby(df["AÑO"]).sum().loc[lambda s: s > 100].items():
        logger.warning("[%s] U_S_M2 publicado difiere >5%% de DOLARES/M2 en %d filas "
                       "(se usa PRECIO_M2)", y, n)
    return df


def clean_text_columns(df):
    """Unifica BARRIO y ORIENT y elimina las filas sin barrio.

    Mayúsculas, sin espacios, y con las variantes de escritura corregidas
    (config.BARRIO_FIXES y config.ORIENT_FIXES). CABA tiene 48 barrios.
    """
    for c, fixes in (("BARRIO", BARRIO_FIXES), ("ORIENT", ORIENT_FIXES)):
        # mayúsculas, sin espacios y con las variantes de escritura unificadas
        txt = df[c].astype("string").str.strip().str.upper()
        txt = txt.replace(fixes)
        # en ORIENT hay valores basura (0, SI) que pasan a NaN
        if c == "ORIENT":
            txt = txt.mask(txt.isin(ORIENT_INVALID))
        # texto vacío = dato faltante
        txt = txt.mask(txt == "")
        # vuelve a tipo object, con NaN de numpy donde falta el dato
        df[c] = txt.astype(object).where(txt.notna(), np.nan)

    # sin barrio no hay ubicación posible: se eliminan esas filas
    n0 = len(df)
    df = df.dropna(subset=["BARRIO"])
    _log_rows("Sin BARRIO", n0, len(df))
    df["COMUNA"] = pd.to_numeric(df["COMUNA"], errors="coerce")
    # control: si no dan 48, quedó alguna variante de escritura sin unificar
    n_barrios = df["BARRIO"].nunique()
    logger.info("BARRIOS distintos: %d (se esperan 48)", n_barrios)
    if n_barrios != 48:
        logger.warning("Cantidad de barrios distinta de 48: revisar variantes de escritura")
    return df


def clean_coordinates(df):
    """Pasa a NaN las coordenadas que caen fuera del rectángulo de CABA.

    Incluye la geocodificación fallida de 2012 (LAT -89.9982 / LON 137.4195).
    """
    for c in ["LAT", "LON"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # el rectángulo de CABA está en config.CABA_LAT y config.CABA_LON
    ok = df["LAT"].between(*CABA_LAT) & df["LON"].between(*CABA_LON)
    bad = df["LAT"].notna() & ~ok
    for y, n in bad.groupby(df["AÑO"]).sum().loc[lambda s: s > 0].items():
        logger.warning("[%s] %d coordenadas inválidas pasan a NaN", y, n)
    # se anulan LAT y LON juntas (una sin la otra no sirve); el Pipeline las imputa después
    df.loc[bad, ["LAT", "LON"]] = np.nan
    return df


def _binary_flag(s: pd.Series) -> pd.Series:
    """Convierte texto sucio a 1.0 (tiene la comodidad) o 0.0 (no la tiene).

    Cuenta como 1: "SI", "si", "S", "SÍ" y cualquier número mayor a 0 (por ejemplo "2").
    Todo lo demás (vacío, 0, "OPCIONAL") es 0.
    """
    txt = s.astype("string").str.strip().str.upper()
    num = pd.to_numeric(txt, errors="coerce")
    # texto que empieza con S / SI / SÍ, o un número mayor a 0
    si = txt.str.match(r"^S[IÍ]?\b", na=False) | (num > 0).fillna(False)
    return pd.Series(np.where(si.to_numpy(dtype=bool), 1.0, 0.0), index=s.index)


def clean_amenities(df):
    """BAULERA, COCHERA, LAVADERO y TERRAZA pasan a 0/1.

    Si una variable no existe en un año (toda la columna vacía ese año), queda NaN
    en ese año y no 0: no se sabe si los departamentos tenían o no la comodidad.
    """
    for c in BINARY_COLUMNS:
        # True en los años en que la columna está completamente vacía
        no_disponible = df[c].isna().groupby(df["AÑO"]).transform("all")
        # 0/1 donde hay dato; NaN donde la variable no existe ese año
        df[c] = _binary_flag(df[c]).where(~no_disponible, np.nan)
        años = sorted(df.loc[no_disponible, "AÑO"].unique().tolist())
        logger.info("%s -> 0/1. Sin dato en los años: %s. Proporción de 1: %.1f%%",
                    c, años, 100 * df[c].mean())
    return df


def clean_antiguedad(df):
    """Deja ANTIGUEDAD como número de años, y NaN cuando no hay dato confiable.

    "A ESTRENAR" pasa a 0 (dato real). El 0 numérico se interpreta como "sin dato":
    en varios años (2006, 2007, 2011) más de la mitad de los avisos tenía 0, lo cual no es
    plausible. Limitación: en 2007, 2013, 2015 y 2016 prácticamente no aparece el texto
    "A ESTRENAR", así que los departamentos realmente a estrenar de esos años quedan NaN.
    """
    txt = df["ANTIGUEDAD"].astype("string").str.strip().str.upper()
    # marca las filas que dicen "A ESTRENAR" (texto, no número)
    estrenar = (txt == "A ESTRENAR").fillna(False).to_numpy(dtype=bool)

    # el resto del texto pasa a número; lo que no se puede convertir queda NaN
    ant = pd.to_numeric(txt, errors="coerce").astype("float64")
    n_cero = int((ant == 0).sum())
    ant = ant.mask(ant == 0)            # 0 numérico = sin dato
    ant = ant.mask(estrenar, 0.0)       # "A ESTRENAR" = dato real
    logger.info("ANTIGUEDAD: %d 'A ESTRENAR' -> 0; %d ceros numéricos -> NaN", int(estrenar.sum()), n_cero)

    # valores negativos o mayores a MAX_ANTIGUEDAD (config) son errores de carga
    invalid = (ant < 0) | (ant > MAX_ANTIGUEDAD)
    ant = ant.mask(invalid)
    logger.info("ANTIGUEDAD: %d valores inválidos -> NaN; %d nulos en total",
                int(invalid.sum()), int(ant.isna().sum()))
    df["ANTIGUEDAD"] = ant
    return df


def clean_counts(df):
    """AMBIENTES y BAÑOS: los valores <= 0 o mayores al máximo (config) pasan a NaN.

    Un 0 se trata como dato faltante y no como monoambiente: la superficie mediana con
    0 ambientes (60 m²) se parece a la del resto (66 m²) y no a la de 1 ambiente (36 m²).
    En BAÑOS, el texto "SI" también pasa a NaN al convertir a número.
    """
    for col, max_val in (("AMBIENTES", MAX_AMBIENTES), ("BAÑOS", MAX_BANOS)):
        x = pd.to_numeric(df[col], errors="coerce")
        invalid = (x <= 0) | (x > max_val)
        logger.info("%s: %d valores (<=0 o >%d) pasan a NaN", col, int(invalid.sum()), max_val)
        df[col] = x.mask(invalid)
    return df


def create_target(df):
    """Crea SEGMENTO (bajo / medio / alto): terciles de PRECIO_M2 calculados dentro de cada año.

    Se calcula por año porque el precio por m² cambia mucho entre 2001 y 2016 (por ejemplo,
    un valor alto en 2003 puede ser bajo en 2013). Así el segmento es relativo al mercado de
    ese año y las tres clases quedan balanceadas (un tercio cada una).
    """
    # posición relativa de cada aviso dentro de su año (entre 0 y 1); method="first" desempata
    pct = df.groupby("AÑO")[TARGET_BASE].rank(pct=True, method="first")
    # 0 a 1/3 = bajo, 1/3 a 2/3 = medio, 2/3 a 1 = alto (etiquetas de texto)
    df[TARGET_NAME] = pd.cut(pct, bins=[0, 1/3, 2/3, 1.0], labels=SEGMENT_LABELS,
                             include_lowest=True).astype(str)
    logger.info("Distribución del target:\n%s", df[TARGET_NAME].value_counts().to_string())
    return df


def clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aplica todos los pasos de limpieza y devuelve el dataset listo para guardar en SQLite."""
    logger.info("=== Inicio de limpieza: %s filas, %s columnas ===", *df_raw.shape)
    df = df_raw.copy()
    # el orden importa: por ejemplo, PRECIO_M2 se crea antes que el target
    for step in (drop_unused_columns, drop_duplicates, clean_prices_and_surface,
                 clean_text_columns, clean_coordinates, clean_amenities,
                 clean_antiguedad, clean_counts, create_target):
        df = step(df)

    # índice 0..n-1: el id_fila que se guarda en MongoDB corresponde a esta posición
    df = df.reset_index(drop=True)
    # comprobaciones finales: si fallan, se corta en lugar de seguir con datos inconsistentes
    assert df[TARGET_NAME].notna().all(), "Hay filas sin target"
    assert df[["M2", "DOLARES", "U_S_M2", "BARRIO"]].notna().all().all(), "Nulos en columnas clave"
    logger.info("=== Fin de limpieza: %s filas, %s columnas ===", *df.shape)
    return df
