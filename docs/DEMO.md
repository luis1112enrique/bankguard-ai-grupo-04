# Guion reproducible de la demo — Grupo 04

Este documento es el material de apoyo para presentar BankGuard AI. No es una presentación de diapositivas ni indica que exista un archivo PPT/PDF en la entrega.

## Archivos necesarios

- [pyproject.toml](../pyproject.toml) y [uv.lock](../uv.lock): entorno y versiones.
- [CLI](../agents/10_bankguard_agent.py) y [paquete BankGuard](../agents/bankguard/): código ejecutable.
- [Transacciones](../agents/data/bank_transactions.csv), [perfiles](../agents/data/client_profiles.csv) y [reglas](../agents/bankguard/risk_config.json): entrada reproducible.
- [index.html](../index.html): frontend de la beta.
- [Plantilla de entorno](../.env.example) y [corpus local](../agents/rag_documents/): configuración de referencia para RAG. La integración necesita además un índice de Pinecone preparado con embeddings compatibles.
- [Arquitectura](ARCHITECTURE.md): diagrama breve para explicar el flujo.

Los comandos siguientes se ejecutan desde la raíz del [repositorio del equipo](https://github.com/luis1112enrique/bankguard-ai-grupo-04). Instala primero Python 3.12 o posterior y uv.

## Preparación

```bash
git clone https://github.com/luis1112enrique/bankguard-ai-grupo-04.git
cd bankguard-ai-grupo-04
uv sync --locked --dev
LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
uv run pytest tests/bankguard -q -m "not integration"
```

La prueba de integración externa queda excluida. Si se presenta RAG real, prepara antes Ollama, descarga `qwen3.5:4b` y `bge-large`, y configura en `.env` las variables de Pinecone descritas en el [README](../README.md). No muestres el contenido del `.env` durante la grabación o presentación.

Los datos incluyen octubre completo de 2026 como escenario simulado. Usa fechas explícitas para que la demo sea independiente de cuándo se ejecute.

Si el entorno es nuevo y se demostrará el RAG, prepara la carga antes de la presentación: configura una clave de Pinecone y un índice propio en `.env`, inicia Ollama con `bge-large` disponible y conserva `DELETE_STALE_VECTORS=false` de la plantilla. Ejecuta explícitamente:

```bash
uv run python agents/9_agentic_rag.py
```

La ingestión escribe en Pinecone y puede crear el índice, consumiendo cuota del proveedor según el plan. No se ejecuta como parte del arranque de la beta. Omite este paso si el corpus ya está preparado o si solo presentarás los modos sin IA.

## 1. Problema y objetivo del Data Steward

Explica que BankGuard ayuda a revisar la calidad y el significado de datos transaccionales, calcular señales de riesgo trazables y recuperar la procedencia de documentos PLD/FT. Muestra el [diagrama](ARCHITECTURE.md) y señala la separación entre cálculos deterministas y redacción del LLM.

Presenta a los integrantes del Grupo 04:

- David Isaí Díaz Avilés.
- Ian Paulo García González.
- Luis Enrique Hernández Torres.
- Pablo Alberto Ramos Roldán.

## 2. Caso determinista sin servicios externos

Ejecuta:

```bash
uv run python agents/10_bankguard_agent.py \
  --client C003 \
  --start 2026-10-01 \
  --end 2026-10-31 \
  --no-compliance
```

Muestra en la salida JSON:

- Cliente y periodo solicitados.
- Conteo, montos y moneda de las operaciones.
- Score, nivel y los cuatro factores calculados.
- Rutas de las gráficas y del expediente, cuando se genera.
- Ausencia de evaluación documental, porque se eligió `--no-compliance`.

Abre las rutas PNG que aparezcan en la salida dentro de `outputs/charts/`. Revisa el JSON generado en `outputs/alerts/`. Los nombres incluyen identificadores de ejecución; no esperes un nombre fijo ni copies resultados de otra corrida como si fueran nuevos.

No es necesario memorizar valores del score: explica los que devuelve esta ejecución y contrástalos con la configuración. Los pesos y umbrales son simulados y la alerta solamente solicita revisión humana.

Con los CSV y las reglas incluidos, la verificación local del caso `C003` devolvió estos valores de referencia:

| Dato | Valor |
| --- | --- |
| Periodo | 2026-10-01 a 2026-10-31 |
| Operaciones | 28 |
| Monto total | 374266.64 MXN |
| Operaciones atípicas | 13 |
| Score simulado | 91.921526 |
| Nivel | alto |

Estos valores proceden del cálculo determinista; el identificador de ejecución y los nombres de artefactos pueden cambiar. Si se modifican los datos o las reglas, vuelve a comprobar las cifras.

## 3. Caso sin historial suficiente

```bash
uv run python agents/10_bankguard_agent.py \
  --client C006 \
  --start 2026-10-01 \
  --end 2026-10-31 \
  --no-compliance \
  --no-charts
```

Muestra el nivel `no_concluyente` y las razones de insuficiencia. Explica que el sistema conserva la incertidumbre y evita clasificar automáticamente como bajo un caso sin historial suficiente.

## 4. Interfaz sin backend

En una terminal:

```bash
uv run python -m http.server 8765 --bind 127.0.0.1
```

Abre [http://127.0.0.1:8765](http://127.0.0.1:8765). Con **Demostración** seleccionado, pulsa «¿Qué es Data Stewardship?» y escribe «Explícalo con más detalle».

Aclara que estas respuestas están predefinidas dentro del HTML. Este paso presenta el diseño de interacción y no constituye una ejecución del LLM, del motor de riesgo ni del RAG. La pregunta sobre clientes/cuentas/transacciones explica un modelo conceptual; el dataset de esta beta no contiene una tabla de cuentas.

## 5. Agentes y evidencia documental, si están configurados

Verifica los modelos disponibles:

```bash
ollama list
```

Con Ollama activo y Pinecone configurado, inicia la API en otra terminal:

```bash
uv run uvicorn agents.bankguard.api:app --host 127.0.0.1 --port 8000
```

Puedes comprobar que el servidor HTTP responde:

```bash
curl --fail http://127.0.0.1:8000/health
```

La respuesta de salud no prueba la conexión con los modelos ni con Pinecone. Conserva abierto el servidor del frontend y selecciona **Conectado**. Escribe, por separado:

```text
Lista los clientes ficticios y la fecha de referencia del dataset.
```

```text
Analiza las operaciones del cliente C003 del 1 al 31 de octubre de 2026 y calcula sus indicadores.
```

```text
¿Qué fuentes del corpus describen el enfoque basado en riesgo para PLD/FT en instituciones bancarias?
```

En la consulta documental muestra las citas devueltas y explica que el especialista debe recuperar evidencia antes de formular afirmaciones normativas. Identifica las limitaciones de fecha del corpus: las disposiciones contienen un texto histórico y las guías no sustituyen una versión legal consolidada.

Para presentar la salida estructurada con cumplimiento también puedes ejecutar:

```bash
uv run python agents/10_bankguard_agent.py \
  --client C003 \
  --start 2026-10-01 \
  --end 2026-10-31 \
  --question "¿Qué criterios PLD/FT documentados son relevantes para revisar este patrón?"
```

No presentes como exitosa una respuesta sin evidencia pertinente. `insufficient_evidence`, `requires_review` o un error indican las limitaciones de la consulta. Si los servicios no están disponibles, conserva la demostración numérica de los pasos anteriores y explica qué parte dependía de ellos.

La prueba optativa de integración real se ejecuta con:

```bash
RUN_BANKGUARD_INTEGRATION=1 uv run pytest tests/bankguard -q -m integration
```

## 6. Reproducibilidad y entrega Git

Muestra los CSV y [risk_config.json](../agents/bankguard/risk_config.json), el [README del proyecto](../README.md) y el [diagrama](ARCHITECTURE.md). La generación de otra copia de los datos se puede demostrar con:

```bash
uv run python -m agents.bankguard.data_generator --output-dir outputs/demo-data
```

El generador no reemplaza archivos existentes por defecto. Usa una carpeta nueva si ya ejecutaste ese comando.

Muestra el historial real de la rama entregada:

```bash
git status --short
git log --oneline -12
git rev-parse HEAD
```

Identifica el commit de la versión presentada y comparte la liga del [repositorio del Grupo 04](https://github.com/luis1112enrique/bankguard-ai-grupo-04). Los commits originales del curso documentan la base del proyecto; no acreditan contribuciones de los integrantes de BankGuard.

Finaliza con las limitaciones del prototipo: datos ficticios, score simulado, revisión humana, corpus histórico y falta de controles de producción.
