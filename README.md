# Taller integrador: clasificación del segmento de valor de departamentos en venta (CABA)

Proyecto final de **Programación Avanzada en Ciencia de Datos** (Licenciatura en Ciencia de Datos).

Pipeline de punta a punta: unifica 16 años (2001-2016) de publicaciones de departamentos en venta de la Ciudad Autónoma de Buenos Aires (CABA), realiza una limpieza exhaustiva y corrección de coordenadas erróneas o fuera de CABA, entrena y optimiza modelos de aprendizaje automático supervisado (Regresión Logística, Random Forest y XGBoost) con scikit-learn, y persiste los datos, resultados, métricas y logs en una **arquitectura híbrida** que combina almacenamiento relacional (SQLite) y documental NoSQL (MongoDB Atlas).

Incluye análisis exploratorio, validación out-of-time, auditoría de data leakage, visualizaciones analíticas, un notebook interactivo de demostración de bases de datos para la defensa oral y un dashboard interactivo en Streamlit.

- **Autor/es:** Luciano Asís, Gustavo Barrajón, Fernando Galera

---

## Qué hace

1. **Carga y unifica** los 16 CSV anuales resolviendo heterogeneidad de nombres de columnas, problemas de encoding, coordenadas geográficas invertidas y formatos de datos disímiles.
2. **Limpia** los datos y crea la variable objetivo `SEGMENTO` (`bajo`, `medio`, `alto`): calculada a partir de los terciles de `PRECIO_M2 = DOLARES / M2` dentro de cada año, de modo que el target sea relativo al mercado de ese año.
3. **Guarda los datos limpios** en SQLite (tabla `datos_limpios`), garantizando consistencia relacional e indexando por `id_fila`.
4. **Entrena y optimiza** clasificadores supervisados mediante un `Pipeline` de scikit-learn (imputación por mediana con indicadores de faltante, One-Hot Encoding con frecuencia mínima y escalado):
   - **Línea Base (`dummy`):** Clasificador sin aprendizaje (`DummyClassifier`) para establecer la cota inferior.
   - **Regresión Logística (`logreg`):** Optimización con `GridSearchCV`.
   - **Random Forest (`random_forest`):** Optimización con `RandomizedSearchCV`.
   - **XGBoost (`xgboost`):** Implementado mediante un wrapper (`XGBTexto`) para admitir etiquetas de texto y optimizado con `RandomizedSearchCV`.
5. **Evalúa rigurosamente** mediante Accuracy, Precision macro, Recall macro, F1-Score macro y matrices de confusión bajo dos esquemas de validación:
   - **Partición Aleatoria:** Split 80/20 estratificado sobre todo el período (2001-2016).
   - **Partición Temporal (Out-of-Time):** Entrenamiento en ventana histórica (2001-2013) y evaluación en años no observados (2014-2016).
   - **Validación Cruzada Estratificada:** 5 folds sobre una muestra de 40.000 filas del conjunto de entrenamiento, para estimar estabilidad y desvíos.
6. **Audita Data Leakage** mediante un experimento comparativo controlado que demuestra cómo se infla artificialmente el rendimiento si se filtran variables derivadas del precio o target (data leakage) (`DOLARES`, `PRECIO_M2`, `AÑO`).
7. **Persiste en MongoDB Atlas** las predicciones individuales vinculadas por `id_fila`, métricas globales por escenario, resultados de validación cruzada, hiperparámetros óptimos, pruebas de leakage y registros de logs de ejecución y eventos.
8. **Visualiza analíticamente** mediante matplotlib y seaborn en `notebooks/03_visualizacion.ipynb`, exportando figuras de alta resolución a `reports/figures/`, y mediante un dashboard interactivo en Streamlit.
9. **Demuestra la integración de ambas bases de datos (SQLite y MongoDB)** en `notebooks/04_demo_bases.ipynb`: consultas analíticas en SQL y MongoDB, cruce híbrido entre ambas bases para auditoría de predicciones, y demostración de operaciones CRUD (crear, leer, actualizar, borrar) con limpieza automática para no alterar los datos limpios.

Todos los módulos emiten logs estructurados a tres destinos simultáneos: consola, archivo rotativo local (`logs/pipeline.log`) y colección documental en MongoDB Atlas (`logs`).

---

## Bases de datos (Integración SQLite + MongoDB Atlas)

El proyecto implementa una integración híbrida relacional y no relacional que aprovecha las fortalezas de cada paradigma:

| Componente | Motor y Ubicación | Contenido y Rol Técnico |
|---|---|---|
| **Tabla: Datos de entrada** | SQLite: `data/processed/deptos.db`, tabla `datos_limpios` | Almacenamiento relacional estructurado de publicaciones limpias con clave primaria `id_fila`. |
| **Colección: Predicciones** | MongoDB Atlas: colección `predicciones` | Documentos con `id_fila` (vínculo relacional mediante id_fila), clase real, clase predicha, probabilidades por segmento, `modelo`, `escenario` y `run_id`. |
| **Colección: Métricas globales** | MongoDB Atlas: colección `metricas` | Exactitud, Precision macro/weighted, Recall macro/weighted, F1-Score y matriz de confusión por modelo y escenario. |
| **Colección: Validación cruzada** | MongoDB Atlas: colección `validacion_cruzada` | Resultados de cada fold por modelo (la media ± desvío estándar se calcula con una consulta de agregación). |
| **Colección: Parámetros y configuración** | MongoDB Atlas: colección `parametros` | Hiperparámetros óptimos resultantes de la búsqueda (`best_params_`), puntuación de validación cruzada y tiempos de cómputo. |
| **Colección: Auditoría de Data Leakage** | MongoDB Atlas: colección `leakage` | Resultados del experimento de contaminación y deltas de sobreestimación métrica. |
| **Colección: Logs de trazabilidad** | MongoDB Atlas: colección `logs` (y archivo `logs/pipeline.log`) | Trazabilidad de logs de ejecución y eventos, comprobaciones de integridad y fallos. |
| **Entorno de demostración** | SQLite y MongoDB: tabla y colección transitoria `demo_crud` | Entorno de práctica seguro para demostración de operaciones CRUD (crear, leer, actualizar, borrar) con limpieza automática para no alterar los datos limpios. |

> **Cruce entre SQLite y MongoDB:** Mediante el identificador unívoco `id_fila`, es posible cruzar directamente cualquier predicción o error clasificado en MongoDB Atlas con los atributos inmobiliarios físicos originales (superficie, ambientes, coordenadas, barrio) persistidos en SQLite.

---

## Estructura del proyecto

```
taller_integrador/
├── config.py                 # Rutas, listas de variables, parámetros de búsqueda y conexión a bases
├── main.py                   # Orquestador del pipeline automatizado con Papermill
├── requirements.txt          # Dependencias principales del proyecto (Windows / Linux)
├── requirements_mac.txt      # Dependencias adaptadas para entornos macOS (Apple Silicon / Intel)
├── .env                      # Credenciales de conexión a MongoDB Atlas (excluido de Git)
├── data/
│   ├── raw/                  # Archivos CSV anuales 2001-2016 (descarga manual)
│   └── processed/            # Base de datos relacional SQLite (deptos.db)
├── src/
│   ├── logger_setup.py       # Configuración jerárquica de logging (consola, archivo y Atlas)
│   ├── mongo_log_handler.py  # Handler de logging con buffer hacia MongoDB
│   ├── data_loader.py        # Lectura, mapeo y unificación de los 16 CSV anuales
│   ├── cleaning.py           # Limpieza de los datos y cálculo de los terciles del target
│   ├── db_sql.py             # Operaciones con SQLite (guardado y lectura de tabla datos_limpios)
│   ├── features.py           # Pipelines de preprocesamiento scikit-learn y particionadores
│   ├── models.py             # Modelos (LogReg, Random Forest, XGBoost), búsqueda de hiperparámetros y validación cruzada
│   ├── evaluation.py         # Métricas de clasificación, matrices de confusión y sobreajuste
│   ├── temporal.py           # Split y entrenamiento en escenario de validación temporal
│   ├── leakage_check.py      # Experimento para detección y auditoría de data leakage
│   ├── db_mongo.py           # Conexión y persistencia en MongoDB Atlas (colecciones de resultados, métricas y logs)
│   └── visualization.py      # Funciones gráficas y visualizaciones analíticas
├── notebooks/
│   ├── 01_eda.ipynb          # Exploración, saneamiento y persistencia en SQLite
│   ├── 02_modelado.ipynb     # Entrenamiento, tuning, validación cruzada y guardado en Atlas
│   ├── 03_visualizacion.ipynb# Visualizaciones diagnósticas y exportación de figuras
│   ├── 04_demo_bases.ipynb   # Demostración interactiva de bases de datos para defensa oral
│   └── ejecutados/           # Cuadernos generados con resultados por Papermill
├── app/
│   └── dashboard.py          # Dashboard interactivo con Streamlit
├── models/                   # Serialización de modelos y resultados entrenados (.joblib)
├── reports/
│   └── figures/              # Figuras exportadas en alta resolución (PNG)
└── logs/
    └── pipeline.log          # Archivo de registro local de eventos y comprobaciones
```

---

## Requisitos

- **Python:** desarrollado y probado con Python 3.13 en adelante
- **MongoDB Atlas:** Cluster M0 gratuito configurado con usuario, contraseña y permisos de red habilitados.
- **Dataset:** Los 16 archivos CSV anuales correspondientes a las publicaciones de departamentos en venta (2001 a 2016).

---

## Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone <URL-DEL-REPOSITORIO>
   cd taller_integrador
   ```

2. **Crear y activar el entorno virtual:**
   ```bash
   python -m venv .venv

   # En Windows:
   .venv\Scripts\activate

   # En Linux / macOS:
   source .venv/bin/activate
   ```

3. **Instalar dependencias:**
   - **En Windows o Linux:**
     ```bash
     pip install -r requirements.txt
     ```
   - **En macOS (Apple Silicon o Intel):**
     Utilizar el archivo adaptado que sustituye componentes específicos de Windows por equivalentes nativos Unix:
     ```bash
     pip install -r requirements_mac.txt
     ```

4. **Registrar el kernel para Jupyter:**
   ```bash
   python -m ipykernel install --user --name=taller_integrador
   ```
   *(El nombre `taller_integrador` debe coincidir con la configuración de kernel definida en `main.py`).*

> **Nota sobre codificación en Windows:** Si regenerás `requirements.txt` en Windows mediante PowerShell con el operador `>`, asegurate de guardarlo en UTF-8 (`pip freeze | Out-File -Encoding utf8 requirements.txt`), evitando UTF-16 que provoca errores de lectura en `pip install`.

---

## Configuración del entorno

### 1. Datos crudos (`data/raw/`)
Descargar manualmente los archivos anuales en formato CSV desde el portal oficial [BA Data - Departamentos en venta](https://data.buenosaires.gob.ar/dataset/departamentos-venta) (el portal bloquea las descargas automáticas) y ubicarlos en `data/raw/` siguiendo el patrón:
```
departamentos-en-venta-2001.csv
departamentos-en-venta-2002.csv
...
departamentos-en-venta-2016.csv
```

### 2. Conexión a MongoDB Atlas (`.env`)
Crear un archivo `.env` en la raíz de `taller_integrador/` con las credenciales de conexión:
```ini
MONGO_URI=mongodb+srv://USUARIO:PASSWORD@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGO_DB=taller_integrador
```
- Sustituir `USUARIO` y `PASSWORD` por las credenciales reales de Atlas, sin `< >`, sin comillas y sin espacios.
- Si la contraseña incluye caracteres especiales, codificarlos con `urllib.parse.quote_plus`.
- El archivo `.env` contiene información sensible y se encuentra excluido del control de versiones mediante `.gitignore`.
- Si no hay conexión con Atlas, el envío de logs a Mongo se desactiva y los logs quedan solo en `logs/pipeline.log` y omite de forma segura las operaciones en la nube.

---

## Guía de ejecución

### 1. Ejecución automatizada del pipeline (`main.py`)
Desde el directorio del proyecto con el entorno virtual activado:

```bash
# Modo prueba rápida (Smoke Test): Regresión Logística y XGBoost con búsqueda mínima, sin escritura en Atlas
python main.py --rapido

# Corrida completa: tuning de LogReg, Random Forest y XGBoost, escenarios, validación cruzada y persistencia
python main.py

# Reutilización de modelos: usa los estimadores ya entrenados en models/results.joblib
# y entrena solo los que falten
python main.py --reusar
```

El script utiliza **Papermill** para ejecutar secuencialmente `01_eda.ipynb` y `02_modelado.ipynb`, parametrizando las corridas y guardando los cuadernos ejecutados con sus salidas en `notebooks/ejecutados/`.

### 2. Generación de gráficos (`notebooks/03_visualizacion.ipynb`)
Permite generar el catálogo de gráficos analíticos y guardarlos automáticamente en `reports/figures/`:
- Distribución de precios y terciles de corte por año.
- Matrices de confusión comparativas entre modelos.
- Rendimiento comparativo en validación temporal (out-of-time).
- Importancia de variables por permutación.
- Gráficos de impacto y diagnóstico de data leakage.

### 3. Demostración interactiva de bases de datos (`notebooks/04_demo_bases.ipynb`)
Diseñado especialmente para la **defensa oral del trabajo práctico**:
- **Parte 1 (Consultas analíticas):**
  - **SQL (SQLite):** Conteos temporales, `GROUP BY` por barrio con filtro `HAVING`, y filtros multivariables de propiedades.
  - **NoSQL (MongoDB Atlas):** Consulta de hiperparámetros óptimos en `parametros`, conteos de inferencias por escenario, agregación con `aggregate` para comparar exactitud y F1-Score multimodelo, y auditoría de logs por severidad.
- **Parte 2 (Cruce híbrido relacional/documental):** Cruce entre colecciones de inferencias de MongoDB y datos tabulares físicos de SQLite utilizando `id_fila`, auditando aciertos y confusiones del modelo.
- **Parte 3 (Demostración CRUD):** Demostración de operaciones CRUD (crear, leer, actualizar, borrar) sobre una tabla y colección dedicadas (`demo_crud`), con limpieza automática para no alterar los datos limpios.

### 4. Dashboard interactivo (`app/dashboard.py`)
Para explorar los resultados en una interfaz gráfica web (necesita que el pipeline completo ya se haya ejecutado y el `.env` configurado):

```bash
streamlit run app/dashboard.py
```

El dashboard se estructura en 6 pestañas:
1. **Datos:** exploración del dataset limpio, con filtros por año, barrio y segmento, mapa y porcentaje de nulos.
2. **Modelos:** métricas por escenario, matriz de confusión y parámetros de cada modelo.
3. **Validación cruzada:** resumen media ± desvío y resultados por fold.
4. **Predicciones:** accuracy por año y por barrio, y muestra de predicciones individuales.
5. **Leakage:** resultados de la prueba de fuga de información.
6. **Logs:** visor de los logs almacenados en MongoDB Atlas.

---

## Resultados de referencia

Corrida de referencia con `run_id` 20261115_184600 (test: 34.824 instancias en escenario aleatorio; 28.266 en escenario temporal):

| Modelo | Accuracy (Aleatorio) | F1 Macro (Aleatorio) | Accuracy (Temporal) | F1 Macro (Temporal) |
|---|---|---|---|---|
| **DummyClassifier** (Estratificado) | 0,333 | 0,333 | 0,333 | 0,333 |
| **Regresión Logística** | 0,619 | 0,615 | 0,608 | 0,604 |
| **Random Forest** | 0,678 | 0,678 | 0,641 | 0,641 |
| **XGBoost** | 0,669 | 0,667 | 0,637 | 0,636 |

En la validación cruzada de 5 folds (muestra de 40.000 filas), Random Forest y XGBoost quedan empatados (accuracy de 0,654 y 0,655) y la Regresión Logística en 0,618.

---

## Decisiones de diseño y consideraciones metodológicas

- **Definición del Target relativo:** Segmentación por terciles de `PRECIO_M2` calculada de forma independiente para cada año calendario. Esto evita clasificar años completos como "caros" o "baratos" por la evolución del mercado (por ejemplo, la crisis de 2001-2002 y la recuperación posterior). Los precios son nominales en dólares, sin ajuste por inflación.
- **Cálculo homogéneo de `PRECIO_M2`:** En el dataset original de 2016, la variable publicada `U_S_M2` corresponde a superficie cubierta en lugar de total. Se recalculó uniformemente la métrica sobre la superficie total para asegurar comparabilidad a lo largo de los 16 años.
- **Prevención de Data Leakage:** Las variables de precio directo (`DOLARES`, `U_S_M2`, `PRECIO_M2`) y variables no generalizables (`AÑO`, `CALLE`, `NUMERO`) fueron excluidas del vector de entrenamiento. La prueba controlada de leakage muestra cuánto subiría la exactitud al incluirlas: de 0,647 (modelo real) a 0,689 con `DOLARES` y a 0,855 con `PRECIO_M2` + `AÑO` (Random Forest de 100 árboles sobre una muestra de 40.000 filas).
- **Tratamiento de valores faltantes:** Variables como `BAÑOS`, `ANTIGUEDAD` o amenidades no estaban presentes en todos los años de publicación. Se imputan dentro del `Pipeline` con indicadores de ausencia (`add_indicator=True`), de modo que los modelos puedan usar la no-respuesta como señal.
- **Sobreajuste:** El Random Forest sobreajusta (accuracy de 0,80 en train contra 0,68 en test) pese a que la búsqueda explora `max_depth`, `min_samples_leaf` y `max_features`. XGBoost muestra una brecha mucho menor (0,686 contra 0,669). Queda documentado como limitación del modelo.

---

## Notas de compatibilidad técnica

- **Aviso en Windows (joblib/loky):** Al ejecutar `main.py` puede visualizarse un mensaje `KeyError` proveniente de `joblib.externals.loky.backend.resource_tracker`. En nuestras corridas apareció después de que los resultados se guardaran en las bases, y las corridas terminaron completas; no encontramos una causa confirmada, pero no alteró los modelos ni las bases de datos.
- **Soporte Unix/macOS:** En entornos macOS se recomienda utilizar el archivo `requirements_mac.txt`.

