"""Configuración centralizada del sistema de logging para consola, archivo y MongoDB.

Este módulo provee la fábrica de loggers (`get_logger`) para todo el proyecto:
    1. Consola (StreamHandler): Muestra eventos de nivel INFO en adelante para monitoreo interactivo.
    2. Archivo rotativo (RotatingFileHandler): Escribe en `logs/pipeline.log` a nivel DEBUG,
       limitando el tamaño a 2 MB con rotación de hasta 3 copias históricas.
    3. Base de datos no relacional (MongoLogHandler): Handler singleton conectado a MongoDB Atlas
       que transmite en lotes los eventos a la colección 'logs' identificados por el `RUN_ID`.
"""
import logging
from logging.handlers import RotatingFileHandler

from config import LOGS_DIR, MONGO_URI, MONGO_DB, RUN_ID

FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_mongo_handler = None
_mongo_tried = False


def _get_mongo_handler():
    """Inicializa de forma diferida (lazy) un único handler compartido para MongoDB.

    Evita abrir múltiples conexiones a MongoDB Atlas si varios módulos importan y solicitan
    loggers en el mismo proceso. Si falla la conexión a Atlas o no está configurada la URI,
    el handler se deshabilita de manera segura sin interrumpir el pipeline.
    """
    global _mongo_handler, _mongo_tried
    if not _mongo_tried:
        _mongo_tried = True
        if MONGO_URI:
            from src.mongo_log_handler import MongoLogHandler
            h = MongoLogHandler(MONGO_URI, MONGO_DB, RUN_ID)
            _mongo_handler = h if h.enabled else None
    return _mongo_handler


def flush_mongo_logs():
    """Fuerza el envío inmediato de cualquier registro de log acumulado en el buffer hacia MongoDB.

    Recomendado al finalizar la ejecución de scripts o notebooks para asegurar que
    los últimos mensajes no queden retenidos en memoria antes del cierre del proceso.
    """
    if _mongo_handler:
        _mongo_handler.flush()


def get_logger(name: str) -> logging.Logger:
    """Crea o recupera un logger con formato homogéneo y canales configurados.

    Evita duplicar handlers si el logger ya fue instanciado previamente en la jerarquía.

    Parameters
    ----------
    name : str
        Nombre del módulo que emite los logs (típicamente `__name__`).

    Returns
    -------
    logging.Logger
        Logger listo para emitir mensajes por consola, archivo y MongoDB.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(FORMAT)

    # 1. Consola (stdout) a nivel INFO para no saturar la salida interactiva
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # 2. Archivo local rotativo a nivel DEBUG para diagnóstico forense completo
    file = RotatingFileHandler(LOGS_DIR / "pipeline.log", maxBytes=2_000_000,
                               backupCount=3, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file)

    # 3. Handler distribuido hacia MongoDB Atlas si está disponible
    mongo = _get_mongo_handler()
    if mongo:
        logger.addHandler(mongo)
    return logger
