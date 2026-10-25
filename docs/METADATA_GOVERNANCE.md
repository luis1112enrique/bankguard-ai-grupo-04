# Metadata y gobierno de datos — BankGuard AI

Este documento describe el contrato de datos y los controles implementados en la beta del Grupo 04. La fuente de verdad de las reglas ejecutables es el código de [`agents/bankguard/`](../agents/bankguard/), junto con [`risk_config.json`](../agents/bankguard/risk_config.json).

## Responsabilidad y clasificación

**Owner del prototipo y sus datasets: Grupo 04.** Integrantes: David Isaí Díaz Avilés, Ian Paulo García González, Luis Enrique Hernández Torres y Pablo Alberto Ramos Roldán. El equipo mantiene en conjunto código, datos, metadata y documentación; no se atribuyen responsabilidades individuales que el equipo no haya acordado.

| Activo | Clasificación y uso |
| --- | --- |
| Transacciones y perfiles incluidos | Datos completamente sintéticos, publicables para la demo académica. Los clientes y sus comportamientos son ficticios; no proceden de expedientes bancarios reales. |
| Corpus en `agents/rag_documents/` | Textos normativos y guías de referencia para recuperación. Deben revisarse su procedencia, versión y vigencia antes de una interpretación aplicable a un caso real. |
| `risk_config.json` | Reglas educativas versionadas. Sus pesos y umbrales no son límites oficiales de una autoridad. |
| Gráficas, resultados y expedientes | Artefactos derivados de la ejecución. Se conservan localmente en `outputs/`; no son reportes regulatorios. |
| Credenciales y archivos `.env` | Secretos de configuración local; deben quedar fuera de Git. Solo las plantillas `.env.example` sin valores reales forman parte de la entrega. |

La beta trabaja exclusivamente con el dataset ficticio entregado. No implementa controles suficientes para incorporar información personal o bancaria real.

## Inventario y reproducibilidad

| Archivo | Contenido y procedencia |
| --- | --- |
| [`agents/data/bank_transactions.csv`](../agents/data/bank_transactions.csv) | 202 operaciones sintéticas de seis clientes (`C001`–`C006`). |
| [`agents/data/client_profiles.csv`](../agents/data/client_profiles.csv) | Seis perfiles y descripciones de los escenarios ficticios. |
| [`agents/bankguard/data_generator.py`](../agents/bankguard/data_generator.py) | Generador reproducible, semilla predeterminada `20261020`. No sobrescribe archivos existentes salvo que se indique `--overwrite`. |
| [`agents/bankguard/risk_config.json`](../agents/bankguard/risk_config.json) | Configuración de factores, umbrales y suficiencia histórica, `schema_version=1`. |
| [`agents/9_agentic_rag.py`](../agents/9_agentic_rag.py) | Preparación e ingestión del corpus normativo en Pinecone; se ejecuta por separado. |

El historial sintético abarca del `2026-07-01` al `2026-09-30`; el periodo principal de análisis es del `2026-10-01` al `2026-10-31`. Estas fechas describen escenarios simulados. Las consultas relativas usan por defecto la fecha máxima del CSV (`2026-10-31`) para mantener reproducible la demostración.

## Diccionario: transacciones

Todos los campos de esta tabla son obligatorios para el motor. El formato de entrega usa CSV UTF-8 y punto decimal.

| Columna | Tipo lógico / ejemplo | Significado y control implementado |
| --- | --- | --- |
| `transaction_id` | Texto, `TX-2026-0001` | Identificador único de operación. Se rechazan vacíos y duplicados. |
| `client_id` | Texto, `C003` | Relación con `client_profiles.client_id`. Se recorta y normaliza a mayúsculas; debe existir en perfiles. |
| `date` | Fecha, `2026-10-01` | Fecha de operación. Debe ser interpretable como fecha; el motor normaliza a día. |
| `amount` | Número decimal, `8559.67` | Monto en la moneda de la fila. Debe ser numérico, finito y mayor que cero. |
| `currency` | Texto, `MXN` | Moneda de tres letras. Se normaliza a mayúsculas y debe estar permitida en configuración (`MXN`, `USD`). |
| `transaction_type` | Texto, `transferencia` | Tipo de operación, no vacío. Se compara con las categorías esperadas del perfil. |
| `channel` | Texto, `movil` | Canal utilizado, no vacío. Se compara con los canales esperados. |
| `location` | Texto, `Ciudad de Mexico` | Ubicación ficticia, no vacía. Se compara con las ubicaciones esperadas. |

El patrón visual `TX-2026-...` es una convención del generador; el validador no exige esa expresión. Tipo, canal y ubicación admiten categorías nuevas no vacías: una categoría distinta al perfil puede generar una inconsistencia analítica, sin constituir por sí misma un error de calidad o evidencia de delito.

## Diccionario: perfiles

Las listas de valores esperados usan `|` como separador, por ejemplo `movil|web`.

| Columna | Tipo lógico / ejemplo | Significado y control implementado |
| --- | --- | --- |
| `client_id` | Texto, `C003` | Clave única obligatoria. No vacía, recortada y normalizada a mayúsculas. |
| `client_name` | Texto, `Cliente Ficticio 003` | Nombre de exhibición ficticio. Lo incluye el generador; no es un campo obligatorio del cálculo. |
| `segment` | Texto, `personas` o `pyme` | Segmento ilustrativo. No determina el score ni forma parte de las columnas mínimas exigidas. |
| `onboarding_date` | Fecha, `2024-11-20` | Fecha ficticia de alta. Obligatoria y válida; participa en el cálculo de cobertura histórica disponible. |
| `history_available_from` | Fecha, `2026-07-01` | Inicio del historial disponible. Obligatoria y válida; delimita el historial utilizable. |
| `typical_amount` | Número, `3000.0` | Monto habitual esperado. Obligatorio, finito y mayor que cero. |
| `max_typical_amount` | Número, `9000.0` | Límite ilustrativo del perfil para detección de montos atípicos. Debe ser finito, positivo y al menos igual a `typical_amount`. |
| `expected_weekly_frequency` | Número, `1.8` | Frecuencia semanal ilustrativa del perfil, obligatoria, finita y positiva. El factor de frecuencia del score usa el historial observado. |
| `expected_currency` | Texto, `MXN` | Moneda esperada, obligatoria y no vacía. Se usa en la comparación del perfil. |
| `expected_transaction_types` | Lista en texto, `transferencia|pago` | Tipos esperados, campo obligatorio no vacío. |
| `expected_channels` | Lista en texto, `movil|web` | Canales esperados, campo obligatorio no vacío. |
| `expected_locations` | Lista en texto, `Ciudad de Mexico` | Ubicaciones esperadas, campo obligatorio no vacío. |
| `scenario_description` | Texto descriptivo | Etiqueta del escenario preparada por el generador para explicar la demo. No se usa para calcular el score. |

## Reglas de calidad ejecutadas

[`risk_engine.py`](../agents/bankguard/risk_engine.py) valida los CSV antes del análisis:

1. Exige las ocho columnas transaccionales y las diez columnas mínimas del perfil descritas como obligatorias.
2. Rechaza claves vacías, operaciones duplicadas, perfiles duplicados, fechas inválidas y números no finitos o no positivos.
3. Comprueba que toda operación tenga un perfil de cliente conocido.
4. Rechaza monedas no permitidas y clientes con operaciones en más de una moneda, porque no hay conversión cambiaria.
5. Normaliza texto y fechas, y ordena las operaciones por fecha e identificador.
6. Exige periodos inclusivos con fin igual o posterior al inicio y duración máxima configurable de 366 días.

La lista describe controles existentes. No hay un catálogo cerrado de canales o ubicaciones, comprobación de identidad, contraste contra bancos, detección de cuentas duplicadas entre clientes ni validación externa de la veracidad de un perfil. La relación cronológica entre las fechas del perfil tampoco se valida como regla independiente.

## Reglas educativas y suficiencia del score

El score se calcula con pandas y NumPy, sin aritmética delegada al modelo de lenguaje:

```text
score = 0.40 × desviación_histórica
      + 0.25 × frecuencia_atípica
      + 0.20 × concentración_de_montos
      + 0.15 × inconsistencia_de_perfil
```

Cada factor está limitado a `[0, 100]`. La inconsistencia de perfil combina monto (40 %), canal (20 %), ubicación (20 %), tipo (10 %) y moneda (10 %). El score se publica con seis decimales; su clasificación usa ese mismo valor publicado.

| Nivel | Regla predeterminada |
| --- | --- |
| `bajo` | Score menor que 40. |
| `medio` | Score desde 40 y menor que 70. |
| `alto` | Score desde 70 hasta 100. |
| `no_concluyente` | Evidencia insuficiente; score numérico `null`. |

Para una conclusión se requieren al menos tres operaciones en el periodo, ocho operaciones históricas, 30 días de cobertura y los cuatro factores disponibles. El historial mira hasta 92 días anteriores al inicio del periodo. La concentración usa una ventana de tres días. La detección de montos atípicos combina el máximo del perfil con mediana y MAD (`outlier_mad_z=3.5`); la detección de ráfagas usa al menos cuatro operaciones diarias y un multiplicador de tres sobre la tasa histórica.

Los valores se leen de `risk_config.json` o del archivo indicado por `BANKGUARD_RISK_CONFIG`. La configuración valida pesos finitos y no negativos que sumen uno, nombres de factores, umbrales ordenados y ventanas positivas. Los detalles de las fórmulas están en el [README técnico](../agents/bankguard/README.md#indicadores-y-score). Cualquier cambio de reglas debe documentarse en un commit y probarse con los mismos escenarios. Un resultado sin historial suficiente conserva `null`; no se convierte en riesgo bajo.

## Metadata y evidencia del RAG

El corpus local de referencia contiene:

| Archivo | Tipo registrado por el prototipo |
| --- | --- |
| `LIC.txt` | Ley (`law`). |
| `Disposiciones_Articulo_115_LIC.txt` | Disposiciones (`regulation`). |
| `CNBV_Guia_Metodologia_Evaluacion_Riesgos_PLD_FT_2019.txt` | Guía metodológica (`methodology_guide`). |
| `GAFI_Enfoque_Basado_en_Riesgo_Sector_Bancario_2014.txt` | Guía internacional (`international_guidance`). |

La ingestión registra metadata por documento y fragmento cuando está disponible:

| Campos | Uso |
| --- | --- |
| `corpus_id`, `source`, `source_path` | Identificar corpus, archivo y ruta de origen. El corpus esperado es `mx-pld-ft`. |
| `title`, `document_type`, `authority`, `jurisdiction`, `language` | Describir fuente, naturaleza documental y contexto. |
| `publication_date`, `last_reform`, `legal_basis`, `subject` | Contexto registrado en la ingestión; puede faltar según la fuente y no certifica vigencia. |
| `page`, `article`, `provision`, `chapter`, `citation` | Localizar el fragmento y permitir referencias auditables cuando se reconocen en el texto. |
| `file_sha256`, `chunk_sha256` | Identificar contenido y detectar cambios. El hash de archivo de texto se calcula sobre su contenido UTF-8 leído. |
| `chunk_index`, `chunk_count`, `start_index`, `embedding_model`, `schema_version` | Describir segmentación y modelo de embeddings; la ingestión bancaria usa `schema_version=2`. |

[`regulatory_search.py`](../agents/bankguard/regulatory_search.py) consulta en modo lectura, fija el namespace (`PINECONE_NAMESPACE`, por defecto `mx-pld-ft`), añade el filtro `corpus_id` y descarta resultados de otro corpus. No todos los metadatos de ingestión se exponen en cada respuesta: la evidencia conserva los campos admitidos por el recuperador, un extracto y la cita. La similitud de recuperación expresa cercanía del fragmento, no certeza jurídica.

El agente de cumplimiento recibe la pregunta actual y, cuando corresponde, un resumen sintético del caso. Verifica que haya recuperación, evidencia y una respuesta que mencione una fuente recuperada; descarta referencias a artículos o páginas ausentes en esa evidencia. Ante fallas de recuperación, ausencia de evidencia o citas no verificadas devuelve un estado de error, evidencia insuficiente o revisión requerida. Estos controles estructurales no prueban que la interpretación legal sea correcta.

La consulta normal no carga ni actualiza vectores. Ejecutar `agents/9_agentic_rag.py` sí escribe en Pinecone y puede eliminar vectores obsoletos administrados por el corpus según su configuración; esa carga se prepara y ejecuta explícitamente fuera de la demo de lectura.

## Trazabilidad y revisión humana

Cada análisis estructurado conserva un `run_id`, solicitud y periodo normalizados, métricas, moneda, factores y contribuciones, advertencias, errores y trazas de ejecución. La evidencia normativa conserva fuente, cita y extracto. Las gráficas incluyen ruta, fecha UTC y SHA-256; los expedientes JSON incluyen identificador, fecha UTC, operaciones atípicas, resultado de cumplimiento y recomendaciones de revisión.

Un nivel medio, alto o no concluyente crea un expediente con `review_status="pending"`. Un nivel bajo devuelve una evaluación con `review_status="not_triggered"` y no persiste un expediente de alerta. La beta no implementa una bandeja con aprobación, cierre y firma de expedientes. La revisión humana consiste en contrastar datos, examinar contexto y confirmar fuentes oficiales vigentes antes de decidir cualquier actuación.

Las trazas locales filtran nombres y patrones comunes de credenciales; ese filtrado no equivale a anonimizar datos personales. No existe una política automática de retención o borrado de `outputs/`, ni un registro inmutable de auditoría. El equipo debe conservar para la entrega el commit de la versión demostrada y, si comparte evidencia, seleccionar únicamente resultados sintéticos sin secretos. Los cambios se acreditan mediante commits reales del equipo.

## Alcance de la validación

La suite [`tests/bankguard/`](../tests/bankguard/) cubre generación, contrato de datos, score y fronteras, insuficiencia histórica, herramientas, artefactos, trazas, API y respuestas regulatorias con dependencias simuladas. El [workflow](../.github/workflows/bankguard.yml) instala el entorno bloqueado y ejecuta las pruebas sin integración ni credenciales. También verifica por CLI los escenarios sintéticos `C003` (alto) y `C006` (no concluyente) con cumplimiento deshabilitado.

Esto no valida disponibilidad de Ollama o Pinecone, calidad de respuestas de un modelo concreto, pertinencia jurídica, vigencia del corpus ni exactitud predictiva en clientes reales. La prueba de integración requiere habilitación y servicios configurados, y no forma parte del CI normal. El score no está calibrado estadísticamente para producción: sirve para explicar y priorizar revisión en una beta académica.
