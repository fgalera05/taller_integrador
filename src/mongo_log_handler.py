"""Handler personalizado de logging para streaming estructurado hacia MongoDB Atlas.

Este módulo implementa `MongoLogHandler`, una subclase de `logging.Handler`:
    1. Transforma cada LogRecord en un documento BSON estructurado (timestamp UTC, nivel, módulo, mensaje).
    2. Almacena temporalmente los registros en un buffer en memoria para reducir la latencia de red.
    3. Descarga el buffer (insert_many) automáticamente al alcanzar `batch_size` o al finalizar el script
       mediante el hook `atexit`.
    4. Implementa manejo de excepciones no bloqueante: si la conexión con Atlas se interrumpe,
       los errores se capturan sin abortar el flujo principal del programa.
"""
import atexit
import logging
from datetime import datetime, timezone

from pymongo import MongoClient


class MongoLogHandler(logging.Handler):
    """Handler de logging que persiste eventos estructurados en una colección de MongoDB.

    Atributos
    ---------
    run_id : str
        Identificador único de la corrida para trazabilidad en Atlas.
    batch_size : int
        Cantidad máxima de registros en buffer antes de forzar una inserción masiva.
    buffer : list of dict
        Acumulador de eventos en memoria pendientes de inserción.
    enabled : bool
        Indica si la conexión con MongoDB Atlas está activa y operativa.
    """

    def __init__(self, uri: str, db_name: str, run_id: str, collection: str = "logs",
                 batch_size: int = 10, level: int = logging.INFO):
        """Inicializa el handler, verifica conectividad con Atlas y registra el hook de salida.

        Parameters
        ----------
        uri : str
            Cadena de conexión de MongoDB Atlas (obtenida de .env).
        db_name : str
            Nombre de la base de datos destino.
        run_id : str
            Identificador de la ejecución actual.
        collection : str, default='logs'
            Nombre de la colección donde se insertarán los documentos de log.
        batch_size : int, default=10
            Umbral de registros acumulados para envío por lotes.
        level : int, default=logging.INFO
            Nivel mínimo de severidad para registrar en Atlas.
        """
        super().__init__(level)
        self.run_id = run_id
        self.batch_size = batch_size
        self.buffer = []
        self.enabled = True
        try:
            # Timeout corto de 5s para no retrasar el inicio en caso de fallas de red
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            self.col = client[db_name][collection]
        except Exception as e:
            self.enabled = False
            print(f"[MongoLogHandler] Sin conexión a Atlas, logs solo en archivo: {e}")
        # Asegura vaciar el buffer al terminar el intérprete de Python
        atexit.register(self.flush)

    def emit(self, record: logging.LogRecord) -> None:
        """Procesa un registro de log individual, lo añade al buffer y vacía si superó batch_size.

        Parameters
        ----------
        record : logging.LogRecord
            Registro generado por una llamada a logger.info(), logger.warning(), etc.
        """
        if not self.enabled:
            return
        try:
            self.buffer.append({
                "run_id": self.run_id,
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc),
                "level": record.levelname,
                "module": record.name,
                "message": record.getMessage(),
            })
            if len(self.buffer) >= self.batch_size:
                self.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        """Transfiere de forma segura (thread-safe) los registros acumulados en memoria hacia MongoDB."""
        self.acquire()
        try:
            if self.enabled and self.buffer:
                docs, self.buffer = self.buffer, []
                try:
                    self.col.insert_many(docs)
                except Exception as e:
                    print(f"[MongoLogHandler] No se pudieron enviar logs: {e}")
        finally:
            self.release()
