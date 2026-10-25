# TEC AI LangChain — Modern Agents with uv

This repository contains the examples used in the Agentic AI portion of the course.

The current material focuses on the modern LangChain agent API and builds progressively from simple tool use toward specialized agents and multi-agent coordination.

```text
LLM
  ↓
Agent with tools
  ↓
Specialized agents
  ↓
Agent as a tool
  ↓
Maestro / coordinator agent
  ↓
Later: LangGraph, MCP, LangSmith and explicit orchestration
```

The project uses **uv** for dependency management and reproducible installs.

---

## Quick start

### 1. Install uv

Linux / macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Check the installation:

```bash
uv --version
```

### 2. Clone the repository

```bash
git clone https://github.com/ofc-tec/tec-ai-langchain.git
cd tec-ai-langchain
```

### 3. Install the project

This repository includes `pyproject.toml` and `uv.lock`.

```bash
uv sync
```

`uv` creates/manages the local `.venv` and installs the locked dependencies.

The project currently requires Python 3.12 or newer.

You can optionally activate the environment:

```bash
source .venv/bin/activate
```

Or simply run scripts with:

```bash
uv run python path/to/script.py
```

---

## Environment variables

Some examples use Gemini and Tavily.

Create a `.env` file in the project root:

```dotenv
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Never commit real API keys to GitHub.

---

## Current agents

The `agents/` folder contains the current examples:

```text
agents/
├── first_agent.py
├── first_free_agent.py
├── first_gemini_agent.py
├── 2_search web_tool_ agent.py
├── data_agent.py
├── plot_agent.py
├── maestro_agent.py
├── test_agents.py
└── data/
```

The folder will grow during the semester.

---

## 1. First agents

The introductory examples use the modern LangChain API:

```python
from langchain.agents import create_agent
```

A minimal agent looks like:

```python
agent = create_agent(
    model=llm,
    tools=[...],
    system_prompt="You are a helpful assistant."
)
```

The goal is to understand that the agent can decide:

- whether a tool is needed
- which tool to call
- which arguments to send
- whether another tool call is needed after receiving a result

---

## 2. Tools

Tools are normal Python functions exposed with `@tool`.

```python
from langchain_core.tools import tool

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b
```

The function signature becomes part of the tool schema seen by the model.

---

## 3. Data Agent

`agents/data_agent.py` is a small specialist that works with a local CSV dataset.

Its tools allow it to:

- inspect the sales dataset
- return rows and column information
- calculate pandas descriptive statistics

The dataset lives under:

```text
agents/data/
```

Run it with:

```bash
uv run python agents/data_agent.py
```

---

## 4. Plot Agent

`agents/plot_agent.py` is a visualization specialist.

Its tools include:

- bar plot
- scatter plot

Run it with:

```bash
uv run python agents/plot_agent.py
```

The plotting implementation uses independent Matplotlib `Figure` / `Axes` objects so concurrent tool calls do not interfere with each other.

---

## 5. Maestro Agent — first multi-agent example

`agents/maestro_agent.py` coordinates the Data Agent and Plot Agent.

```text
                     Maestro Agent
                    /             \
                   /               \
            Data Agent           Plot Agent
           /         \           /        \
   inspect_csv  summarize_csv  bar_plot  scatter_plot
```

The important idea is that an **agent can itself be wrapped and used as a tool by another agent**.

A request such as:

```text
Analyze the sales dataset, give me descriptive statistics,
create a bar plot of sales by region,
and create a scatter plot of advertising spend versus sales.
```

can produce:

```text
User
  ↓
Maestro
  ↓
Data Agent
  ↓
Dataset values and statistics
  ↓
Maestro
  ↓
Plot Agent
  ↓
bar_plot / scatter_plot
```

The Maestro does not read the CSV directly and does not create plots directly. It delegates to the specialists.

Run it with:

```bash
uv run python agents/maestro_agent.py
```

---

## Testing agents

Test the specialists independently before connecting them:

```bash
uv run python agents/test_agents.py
```

This helps verify each tool and agent separately before introducing coordination.

---

## Why uv?

`uv` keeps the classroom environment reproducible.

Normally students only need:

```bash
uv sync
```

When dependencies change, run `uv sync` again.

The current project includes modern LangChain, LangGraph, Gemini integration, Ollama integration, OpenAI integration, Tavily, pandas, Matplotlib, DuckDB, FastMCP, and Google client libraries through `pyproject.toml`.

---

## Course progression

```text
simple agent
    ↓
agent chooses Python tools
    ↓
agent uses external tools
    ↓
specialized Data Agent
    ↓
specialized Plot Agent
    ↓
Maestro coordinates both agents
    ↓
state / checkpoints
    ↓
LangGraph orchestration
    ↓
MCP tools and external systems
    ↓
LangSmith tracing and evaluation
```

The goal is to introduce the abstractions one at a time rather than beginning with a large framework stack.

---

## Troubleshooting

### `ModuleNotFoundError`

Run scripts from the repository root whenever possible:

```bash
cd tec-ai-langchain
uv run python agents/data_agent.py
```

### Dependencies look out of sync

```bash
uv sync
```

### Matplotlib GUI / Tk errors

Agent tools may execute concurrently. Plotting tools should avoid shared pyplot GUI state and save figures using isolated `Figure` / `Axes` objects.

### API key errors

Check that the required variables exist in your environment or `.env` file.

---

## Legacy material

Earlier versions of this repository contained manual ReAct loops, callbacks, and older LangChain patterns.

Those examples are still useful for understanding classic ReAct mechanics, but the `main` branch now follows the modern LangChain agent API used in the current course.

Older material can remain available in historical branches for reference.

---

## License / Credits

Educational material for Agentic AI coursework at Tecnológico de Monterrey.

Built around LangChain, LangGraph, Gemini, Python, uv, and related open-source tools.
