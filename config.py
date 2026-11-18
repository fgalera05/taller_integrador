"""Configuración central del proyecto.

Acá viven las rutas, los umbrales de limpieza, las listas de columnas, los
hiperparámetros y los nombres de las colecciones de MongoDB. Los demás módulos
(src/*.py) importan estos valores en lugar de repetirlos.

Nota: src/models.py copia algunos valores al importarse, así que si un notebook
los sobrescribe (por ejemplo con Papermill) tiene que hacerlo ANTES de importar
ese módulo.
"""
from pathlib import Path
import os
from datetime import datetime
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Rutas del proyecto (relativas a la carpeta donde está este archivo)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data" / "raw"              # CSV originales (descarga manual)
DATA_PROCESSED = BASE_DIR / "data" / "processed"  # base SQLite con los datos limpios
LOGS_DIR = BASE_DIR / "logs"

# ---------------------------------------------------------------------------
# MongoDB Atlas: las credenciales se leen del archivo .env (que NO se sube a Git)
# ---------------------------------------------------------------------------
load_dotenv(BASE_DIR / ".env")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "taller_integrador")
# Identifica cada ejecución: todo lo que se guarda en Mongo lleva este run_id.
# Se calcula al importar el módulo, así que cada kernel/proceso tiene el suyo.
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------------------------
# Origen de los datos
# ---------------------------------------------------------------------------
# ROOT_URL, RESOURCE_IDS y YEAR_TO_RESOURCE eran para la descarga automática.
# Ya no se usan (el portal la bloquea y los CSV se bajan a mano); se dejan por
# si se quisiera retomar esa descarga.
ROOT_URL = "https://data.buenosaires.gob.ar/dataset/departamentos-venta/resource/juqdkmgo-70"
RESOURCE_IDS = list(range(1, 32, 2))          # 1,3,5,...,31
YEARS = list(range(2001, 2017))               # 2001..2016
YEAR_TO_RESOURCE = dict(zip(YEARS, RESOURCE_IDS))

# Nombre de los archivos en data/raw/ (uno por año)
RAW_FILE_PATTERN = "departamentos-en-venta-{year}.csv"

# Los CSV vienen en UTF-8 y separados por ";". Con "unicode_escape" las tildes
# y la Ñ salían corruptas (BAÑOS quedaba 100% nula).
CSV_PARAMS = dict(encoding="utf-8-sig", sep=";")

# ---------------------------------------------------------------------------
# Unificación de columnas entre años (los nombres cambian de un año a otro)
# ---------------------------------------------------------------------------
COLUMN_RENAME_MAP = {
    "LATITUD": "LAT",
    "LONGITUD": "LON",
    "M2CUB": "M2_CUBIERTOS",
    "BAÃOS": "BAÑOS",   # variante del encoding viejo; ya no aparece con utf-8-sig
    "BAÑO": "BAÑOS",
    "BAUL": "BAULERA",
    "BARRIOS": "BARRIO",
    "A_ESTRENAR": "ES_ESTRENAR",
}

# Columnas que debe tener el dataset unificado (las que faltan en un año se
# rellenan con NaN y quedan registradas en el log)
TARGET_COLUMNS = [
    "AÑO", "CALLE", "NUMERO", "M2", "M2_CUBIERTOS", "DOLARES", "U_S_M2",
    "AMBIENTES", "DORMITORIO", "PISO", "ANTIGUEDAD", "ORIENT", "BAULERA",
    "COCHERA", "BAÑOS", "LAVADERO", "TERRAZA", "BARRIO", "COMUNA",
    "DIRECCION_NORMALIZADA", "LAT", "LON",
]

# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------
# Columnas casi vacías (más del 90% de nulos): se descartan
DROP_COLUMNS = ["M2_CUBIERTOS", "DORMITORIO", "PISO", "DIRECCION_NORMALIZADA"]
# Comodidades que vienen como texto sucio (SI, si, S, números) y pasan a 0/1
BINARY_COLUMNS = ["BAULERA", "COCHERA", "LAVADERO", "TERRAZA"]

# Rectángulo aproximado de CABA: lo que cae afuera es una coordenada inválida
CABA_LAT = (-34.72, -34.52)
CABA_LON = (-58.54, -58.33)

# Umbrales iniciales: ajustarlos mirando el EDA
M2_RANGE = (15, 500)                  # superficie en m²
DOLARES_RANGE = (5_000, 5_000_000)    # precio total en USD
USM2_RANGE = (100, 10_000)            # precio por m² en USD
MAX_AMBIENTES = 10
MAX_BANOS = 8
MAX_ANTIGUEDAD = 120                  # años

# ---------------------------------------------------------------------------
# Variable objetivo y columnas que NO pueden ser predictoras
# ---------------------------------------------------------------------------
TARGET_NAME = "SEGMENTO"
SEGMENT_LABELS = ["bajo", "medio", "alto"]    # orden usado en matrices y gráficos

# El target son terciles de PRECIO_M2 (= DOLARES / M2) calculados dentro de cada año.
# Se recalcula porque en 2016 el U_S_M2 publicado usa los m² cubiertos.
TARGET_BASE = "PRECIO_M2"
# Fuga de información (DOLARES, U_S_M2, PRECIO_M2, el propio target), identificadores
# (CALLE, NUMERO) y AÑO (el target ya es relativo a cada año).
# get_xy() lanza un error si alguna de estas columnas se cuela entre las predictoras.
NON_FEATURES = ["CALLE", "NUMERO", "DOLARES", "U_S_M2", "PRECIO_M2", "SEGMENTO", "AÑO"]

# Variantes de escritura que se unifican
BARRIO_FIXES = {
    "NUNEZ": "NUÑEZ",
    "MONTSERRAT": "MONSERRAT",
    "PARQUE AVELLANED": "PARQUE AVELLANEDA",
    "VILLA GRAL. MITR": "VILLA GRAL. MITRE",
}
ORIENT_FIXES = {"FRENTE Y CONTRA": "FRENTE Y CONTRAFRENTE",
                "DOBLE FRENTE": "FRENTE Y CONTRAFRENTE"}
ORIENT_INVALID = ["0", "SI"]          # valores basura: pasan a NaN

# ---------------------------------------------------------------------------
# Base relacional (Tabla 1: datos limpios)
# ---------------------------------------------------------------------------
SQLITE_PATH = DATA_PROCESSED / "deptos.db"
TABLE_CLEAN = "datos_limpios"

# ---------------------------------------------------------------------------
# Modelado
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5                     # particiones de StratifiedKFold (antes 3)
SCORING = "f1_macro"             # métrica que optimiza la búsqueda de hiperparámetros
SEARCH_SAMPLE_SIZE = 20_000      # filas para buscar hiperparámetros. None = todo el train

# Predictoras: numéricas (se imputan con la mediana y se escalan) y
# categóricas (se imputan con "SIN_DATO" y se codifican one-hot)
NUMERIC_FEATURES = ["M2", "AMBIENTES", "ANTIGUEDAD", "BAÑOS", "BAULERA",
                    "COCHERA", "LAVADERO", "TERRAZA", "LAT", "LON"]
CATEGORICAL_FEATURES = ["BARRIO", "ORIENT"]

# Espacios de búsqueda. El prefijo "clf__" es el nombre del paso del Pipeline.
RF_N_ITER = 5                    # combinaciones que prueba RandomizedSearchCV (Random Forest)
XGB_N_ITER = 8                   # idem para XGBoost
PARAM_SPACES = {
    "logreg": {"clf__C": [0.01, 0.1, 1, 10]},
    "random_forest": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [10, 20, None],
        "clf__min_samples_leaf": [1, 5, 10],
        "clf__max_features": ["sqrt", 0.5],
    },
    "xgboost": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    },
}
# Parámetros fijos de XGBoost (no se buscan). El objetivo multiclase se elige solo
# según la cantidad de clases; "mlogloss" es la pérdida para multiclase.
XGB_FIXED_PARAMS = dict(tree_method="hist", eval_metric="mlogloss", random_state=RANDOM_STATE)

# Informe de validación cruzada (cross_validate con CV_FOLDS particiones)
CV_REPORT_SAMPLE_SIZE = 40_000   # se hace sobre una muestra del train para no tardar horas
CV_REPORT_SCORING = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}

MODELS_DIR = BASE_DIR / "models"     # modelos entrenados (joblib) y caché de resultados

# ---------------------------------------------------------------------------
# Colecciones de MongoDB
# ---------------------------------------------------------------------------
COL_PREDICTIONS = "predicciones"     # Tabla 2: una predicción por fila de test y modelo
COL_METRICS = "metricas"             # Tabla 2: métricas globales y matriz de confusión
COL_PARAMS = "parametros"            # Tabla 3: configuración de cada modelo
COL_LEAKAGE = "leakage"              # prueba de fuga de información
COL_CV = "validacion_cruzada"        # resultados por fold de la validación cruzada

# ---------------------------------------------------------------------------
# Otros
# ---------------------------------------------------------------------------
DUMMY_STRATEGY = "stratified"        # línea base: predice al azar respetando las proporciones
TEMPORAL_CUTOFF = 2013               # escenario temporal: train <= 2013, test >= 2014

FIGURES_DIR = BASE_DIR / "reports" / "figures"