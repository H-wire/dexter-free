# Dexter Agent: Development Guide & Architecture

This document combines repository guidelines, coding standards, and architectural documentation for the Dexter financial research agent.

---

## Table of Contents

1. [Project Structure & Module Organization](#project-structure--module-organization)
2. [Architecture Overview](#architecture-overview)
3. [Core Workflow](#core-workflow)
4. [Key Components](#key-components)
5. [Build, Test, and Development Commands](#build-test-and-development-commands)
6. [Coding Style & Naming Conventions](#coding-style--naming-conventions)
7. [Testing Guidelines](#testing-guidelines)
8. [Commit & Pull Request Guidelines](#commit--pull-request-guidelines)
9. [Security & Configuration Tips](#security--configuration-tips)
10. [Logging & LLM Output](#logging--llm-output)
11. [Data Sources](#data-sources)
12. [Agent Capabilities (Tools)](#agent-capabilities-tools)

---

## Project Structure & Module Organization

```
src/dexter/
├── agent.py              # Main orchestration (Agent class)
├── cli.py                # CLI entry point
├── model.py              # LLM abstraction layer
├── prompts.py            # System prompts (the "brain")
├── schemas.py            # Pydantic models
├── config.py             # Configuration management
├── tools/
│   ├── __init__.py       # Tool factory
│   ├── yf_*.py           # Yahoo Finance tools (default provider)
│   ├── finance/          # FinancialDatasets tools (premium provider)
│   │   ├── api.py        # API client
│   │   ├── prices.py
│   │   ├── fundamentals.py
│   │   ├── metrics.py
│   │   ├── news.py
│   │   ├── filings.py
│   │   └── estimates.py
│   ├── web_search/       # Web search abstraction
│   │   ├── base.py       # BaseSearcher abstract class
│   │   └── duckduckgo.py # DuckDuckGo implementation
│   └── web_search_tool.py # Web search tool
└── utils/
    ├── logger.py         # Logging utilities
    ├── ui.py             # Terminal UI with progress indicators
    └── intro.py          # Welcome banner

Top-level files:
├── pyproject.toml        # Python dependencies & project metadata
├── uv.lock               # Locked dependencies (uv)
├── index.js              # Node wrapper
├── package.json          # Node dependencies
├── env.example           # Example environment variables
├── README.md             # User-facing documentation
├── AGENTS.md             # This file (dev guide & architecture)
└── TODO.md               # Task tracking and improvement plans
```

**Key Principles**:
- User-facing commands: `cli.py`
- Pure data access: `tools/`
- Shared helpers: `utils/`
- Data models: `schemas.py`
- Business logic: `agent.py`

---

## Architecture Overview

### High-Level Summary

Dexter is a sophisticated financial research agent built on the LangChain framework. Its architecture is designed around a **structured, multi-step reasoning process** orchestrated by the `Agent` class in `src/dexter/agent.py`. The agent breaks down a user's query into a series of tasks, executes them, and then synthesizes an answer. This multi-step process, combined with self-correction and dynamic tool use, makes for a robust and powerful research assistant.

**Dexter is essentially "Claude Code, but built specifically for financial research."**

### Architecture Diagram

```
User Query
    ↓
┌─────────────────────┐
│  Planning Agent     │ ← PLANNING_SYSTEM_PROMPT
│  (Task Decomposer)  │
└─────────┬───────────┘
          ↓
    [Task List]
          ↓
    ┌─────────────┐
    │  Task Loop  │ (for each task)
    └─────┬───────┘
          ↓
┌─────────────────────┐
│  Action Agent       │ ← ACTION_SYSTEM_PROMPT
│  (Tool Selector)    │
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  Argument Optimizer │ ← TOOL_ARGS_SYSTEM_PROMPT (unique!)
│  (Refine Params)    │
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  Tool Execution     │ (parallel execution supported)
│  (Financial Data)   │
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  Validation Agent   │ ← VALIDATION_SYSTEM_PROMPT
│  (Task Complete?)   │
└─────────┬───────────┘
          ↓
    Task Complete? ──No──→ [Loop back to Action Agent]
          ↓ Yes
    More Tasks? ──Yes──→ [Next task]
          ↓ No
┌─────────────────────┐
│  Answer Agent       │ ← ANSWER_SYSTEM_PROMPT
│  (Synthesize Final) │
└─────────┬───────────┘
          ↓
    Final Response
```

---

## Core Workflow

The application's core logic resides in the `Agent.run` method within `src/dexter/agent.py`. The workflow is as follows:

### 1. Planning Phase

The user's query is first processed by an LLM using a specialized `PLANNING_SYSTEM_PROMPT` (from `src/dexter/prompts.py`). The goal is to break the query down into a series of discrete, actionable tasks. The output is structured as a `TaskList` schema (from `src/dexter/schemas.py`).

**Example**:
```
User Query: "What is Apple's revenue growth and profitability?"

Tasks Generated:
1. Get Apple's income statements
2. Calculate revenue growth rate
3. Analyze profitability metrics
```

### 2. Execution Loop

The agent iterates through each task identified in the planning phase. For each task, it enters a sub-loop to perform the necessary actions.

### 3. Action Selection

Inside the task loop, the agent consults the LLM again with the `ACTION_SYSTEM_PROMPT`. It provides the LLM with a set of available tools and asks it to choose the next action to perform to accomplish the current task.

**Tool Selection Features**:
- Smart filtering: `_select_tools_for_task()` filters tools based on keywords (e.g., "10-K" → filing tools)
- Context-aware: Only relevant tools are presented to the LLM

### 4. Argument Optimization (Unique Feature!)

In a **unique and crucial step**, if the LLM decides to use a tool, the agent makes *another* LLM call. This time, it uses the `TOOL_ARGS_SYSTEM_PROMPT` to refine and optimize the parameters for the chosen tool. This ensures the tool is called with the most effective arguments.

**Why This Matters**: The LLM might initially select a tool with suboptimal parameters. This optimization step dramatically improves data retrieval quality.

### 5. Tool Execution

The agent executes the specified tool with the newly optimized arguments.

**Execution Features**:
- **Parallel execution**: Multiple tools can run concurrently (ThreadPoolExecutor)
- **Argument expansion**: List arguments automatically expand into multiple parallel calls
- **Argument normalization**: Handles loose inputs (symbol→ticker, latest="true"→True)
- **Error handling**: Graceful failure with context logging

**Safety Mechanisms**:
- Step limits: Global (20) and per-task (5)
- Loop detection: Tracks last 4 actions to prevent infinite loops
- Parallel execution limits: Configurable (default: 4)

### 6. Validation

After executing the tool, the agent performs a validation step. It asks the LLM if the current task is complete, using the `VALIDATION_SYSTEM_PROMPT` and expecting an `IsDone` structured response. If the task is not yet complete, the action loop continues.

### 7. Answer Synthesis

Once all tasks in the main plan are marked as complete, the agent aggregates all the outputs and observations gathered from the tool executions. It sends this collected evidence to the LLM one final time with the `ANSWER_SYSTEM_PROMPT` to generate a comprehensive, user-facing answer.

---

## Key Components

### `src/dexter/agent.py` (516 lines)

The central orchestrator. The `Agent` class manages the entire workflow, from planning to execution and final response generation.

**Key Methods**:
- `run(query)`: Main entry point for agent execution
- `plan_tasks(query)`: Decomposes query into tasks
- `_select_tools_for_task(task)`: Filters relevant tools
- `optimize_tool_args(tool, args, task)`: Refines tool parameters
- `_execute_tool(tool, args)`: Executes single tool
- `_execute_tools_parallel(tool_calls)`: Parallel execution
- `validate_task_completion(task, observations)`: Checks if task is done
- `_generate_answer(query, observations)`: Synthesizes final response
- `_normalize_tool_args(args, tool_name)`: Normalizes parameters
- `_expand_tool_args(args)`: Expands list args into multiple calls

**Configuration**:
```python
Agent(
    max_steps=20,              # Global safety limit
    max_steps_per_task=5,      # Per-task iteration limit
    data_provider="yfinance"   # Provider selection
)
```

### `src/dexter/model.py` (393 lines)

This file provides a critical abstraction layer for all LLM interactions. The `call_llm` function handles model configuration, structured output binding (ensuring the LLM's response conforms to a Pydantic schema), tool binding, and error handling. It's designed for compatibility with various models, including local Ollama instances.

**Key Capabilities**:
- **Multi-provider support**: OpenAI API, Ollama (with JSON mode), local OpenAI-compatible servers
- **Structured output binding**: Uses LangChain's `with_structured_output()` for Pydantic schemas
- **Ollama JSON mode**: Special handling with `extra_body={"format":"json"}`
- **Tool binding**: Attaches tools to LLM for function calling
- **Robust error handling**: Falls back to plain text when structured output fails
- **Empty answer detection**: Retries without structured output if Answer is empty/"None"
- **Connection retry logic**: Exponential backoff for transient API errors
- **Prompt truncation**: Configurable via `DEXTER_MAX_PROMPT_CHARS` to prevent context overflow
- **LLM instance caching**: Reuses ChatOpenAI instances based on config

**Function Signature**:
```python
def call_llm(
    prompt: str,
    system_prompt: str = "",
    output_schema: Optional[type] = None,
    tools: Optional[list] = None
) -> Any:
    """
    Call LLM with structured output and error handling.

    Args:
        prompt: User message
        system_prompt: System instructions
        output_schema: Pydantic schema for structured output
        tools: List of LangChain tools for function calling

    Returns:
        Structured output (if schema provided) or string
    """
```

### `src/dexter/prompts.py` (189 lines)

This file can be considered the **"brain" of the agent**. It contains all the system prompts that guide the LLM's reasoning at each specific step of the workflow. The clarity and detail of these prompts are fundamental to the agent's performance.

**Six Specialized Prompts**:
1. `PLANNING_SYSTEM_PROMPT`: Task decomposition
2. `ACTION_SYSTEM_PROMPT`: Tool selection
3. `TOOL_ARGS_SYSTEM_PROMPT`: Argument optimization
4. `VALIDATION_SYSTEM_PROMPT`: Task completion check
5. `ANSWER_SYSTEM_PROMPT`: Final synthesis
6. Additional prompts for specific scenarios

### `src/dexter/schemas.py` (34 lines)

Defines the Pydantic models (`TaskList`, `IsDone`, `Answer`, etc.) used to enforce a strict, predictable structure on the LLM's outputs. This is essential for making the agent's behavior reliable.

**Key Schemas**:
- `TaskList`: List of tasks from planning phase
- `IsDone`: Boolean for validation
- `Answer`: Final response structure
- `ToolCall`: Tool name and arguments

### `src/dexter/tools/__init__.py` (95 lines)

This file implements the **pluggable tool architecture**. It acts as a factory, providing the correct set of tools based on the selected data provider (e.g., `yfinance` or `financialdatasets`). This separation of concerns makes it easy to extend the agent with new data sources.

**Function**:
```python
def get_tools(provider: str = "yfinance") -> List[BaseTool]:
    """
    Get tools for the specified data provider.

    Args:
        provider: "yfinance", "financialdatasets", or "web"

    Returns:
        List of LangChain tool objects
    """
```

### `src/dexter/cli.py` (66 lines)

The main entry point for the application. It handles command-line arguments, environment setup, and the instantiation and startup of the main `Agent` loop.

**Features**:
- Interactive REPL with `prompt_toolkit`
- Command-line overrides: `--provider`, `--openai-base-url`, `--openai-model`
- Exit commands: "exit" or "quit"
- Welcome banner and UI

### `src/dexter/config.py` (9 lines - to be expanded)

API key configuration. Currently minimal, but planned to be expanded into a comprehensive configuration management system using Pydantic Settings.

**Current**:
```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FINANCIAL_DATASETS_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
```

**Planned**: See TODO.md task #4 for comprehensive Settings class design.

---

## Build, Test, and Development Commands

### Installation

```bash
# Install Python dependencies from pyproject.toml/uv.lock
uv sync

# Install with dev dependencies
uv sync --extra dev
```

### Running the Agent

```bash
# Run CLI locally with default provider (yfinance)
uv run dexter-agent

# Specify provider
uv run dexter-agent --provider yfinance
uv run dexter-agent --provider financialdatasets
uv run dexter-agent --provider web

# Via Node wrapper
node index.js --provider yfinance

# Alternate entry for debugging
uv run python -m dexter.cli --provider yfinance

# With custom model/endpoint
uv run dexter-agent --openai-base-url http://localhost:11434/v1 --openai-model llama2
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src/dexter --cov-report=html --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_yf_shared.py

# Run with verbose output
uv run pytest -v

# Run tests matching pattern
uv run pytest -k "normalization"
```

### Code Quality

```bash
# Type checking (after mypy setup - see TODO.md #6)
mypy src/dexter

# Format code (if black configured)
black src/dexter tests/

# Lint code (if ruff configured)
ruff check src/dexter
```

---

## Coding Style & Naming Conventions

### Python Style

- **Standard**: Follow PEP 8
- **Indentation**: 4 spaces (no tabs)
- **Line Length**: 100 characters max
- **Type Hints**: Required for all function signatures
- **Docstrings**: Required for public functions/classes (Google style)

**Example**:
```python
def calculate_growth_rate(
    initial_value: float,
    final_value: float,
    periods: int
) -> float:
    """
    Calculate compound annual growth rate.

    Args:
        initial_value: Starting value
        final_value: Ending value
        periods: Number of periods

    Returns:
        Growth rate as a decimal (e.g., 0.15 for 15%)

    Raises:
        ValueError: If initial_value is zero or negative
    """
    if initial_value <= 0:
        raise ValueError("Initial value must be positive")

    return (final_value / initial_value) ** (1 / periods) - 1
```

### Naming Conventions

- **Modules**: `lowercase_underscores.py`
- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private methods**: `_leading_underscore`
- **Protected methods**: `_leading_underscore`
- **Dunder methods**: `__double_underscore__` (only for special Python methods)

**Examples**:
```python
# Module
src/dexter/financial_analysis.py

# Functions/Variables
def calculate_revenue_growth(ticker: str) -> float:
    growth_rate = 0.15
    return growth_rate

# Classes
class FinancialAnalyzer:
    pass

# Constants
MAX_RETRIES = 3
DEFAULT_PROVIDER = "yfinance"

# Private methods
class Agent:
    def _normalize_tool_args(self, args: dict) -> dict:
        pass
```

### Code Organization

**Preferred Import Order**:
1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# Standard library
import os
from typing import List, Dict, Optional

# Third-party
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Local
from dexter.model import call_llm
from dexter.schemas import TaskList
from dexter.utils.logger import Logger
```

### Logging and UI

- Use `dexter.utils.logger.Logger` for consistent logging
- Use `dexter.utils.ui.show_progress` decorator for long operations
- Log levels: DEBUG, INFO, WARNING, ERROR
- Avoid print() statements; use logger instead

```python
from dexter.utils.logger import logger
from dexter.utils.ui import show_progress

logger.info("Starting agent execution")
logger.debug(f"Tool args: {args}")

@show_progress("Fetching financial data...", "")
def fetch_data(ticker: str) -> dict:
    # Long-running operation
    return data
```

---

## Testing Guidelines

### Framework

- **Primary**: `pytest`
- **Async**: `pytest-asyncio`
- **Mocking**: `pytest-mock`
- **Coverage**: `pytest-cov`

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_agent.py            # Agent orchestration tests
├── test_model.py            # LLM abstraction tests
├── test_tools.py            # Tool execution tests
├── test_normalization.py    # Argument normalization tests
├── test_security.py         # Security/sanitization tests
└── integration/
    ├── __init__.py
    └── test_end_to_end.py   # Full agent query tests
```

**Mirroring**: Test files should mirror the structure of `src/dexter`
- `src/dexter/tools/yf_prices.py` → `tests/tools/test_yf_prices.py`

### Test Naming

- **Files**: `test_*.py`
- **Functions**: `test_<feature>_<scenario>()`
- **Classes**: `Test<Feature>`

**Examples**:
```python
# Good test names
def test_normalize_period_converts_3_years_to_3y():
    pass

def test_validate_task_completion_returns_false_on_llm_error():
    pass

def test_execute_tools_parallel_respects_max_concurrency():
    pass

# Bad test names
def test_normalize():
    pass

def test_function():
    pass
```

### Testing Best Practices

1. **No External API Calls**: Use mocks/fixtures for all networked operations
2. **Isolated**: Tests should not depend on each other
3. **Fast**: Unit tests should run in milliseconds
4. **Deterministic**: No random or time-dependent behavior
5. **Coverage**: Aim for >70% code coverage

**Example Test**:
```python
import pytest
from unittest.mock import Mock, patch
from dexter.agent import Agent

def test_normalize_tool_args_converts_symbol_to_ticker():
    """Test that 'symbol' parameter is normalized to 'ticker'."""
    agent = Agent()

    args = {"symbol": "AAPL", "period": "annual"}
    result = agent._normalize_tool_args(args, "yf_get_income_statements")

    assert "ticker" in result
    assert result["ticker"] == "AAPL"
    assert "symbol" not in result

@patch('dexter.model.call_llm')
def test_plan_tasks_returns_task_list(mock_call_llm):
    """Test that planning phase returns structured TaskList."""
    mock_call_llm.return_value = Mock(tasks=["Task 1", "Task 2"])

    agent = Agent()
    tasks = agent.plan_tasks("What is Apple's revenue?")

    assert len(tasks) == 2
    assert mock_call_llm.called
```

### Fixtures

Create shared fixtures in `conftest.py`:

```python
# tests/conftest.py
import pytest
from dexter.agent import Agent

@pytest.fixture
def agent():
    """Provide a test agent instance."""
    return Agent(max_steps=5, data_provider="yfinance")

@pytest.fixture
def sample_task():
    """Provide a sample task for testing."""
    return "Get Apple's income statement"

@pytest.fixture
def mock_tool_response():
    """Provide a mock tool response."""
    return {
        "ticker": "AAPL",
        "revenue": 394328000000,
        "net_income": 99803000000
    }
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_agent.py

# Run specific test
uv run pytest tests/test_agent.py::test_normalize_tool_args

# Run with coverage
uv run pytest --cov=src/dexter --cov-report=term-missing

# Run integration tests only
uv run pytest tests/integration/

# Run in parallel (with pytest-xdist)
uv run pytest -n auto
```

---

## Commit & Pull Request Guidelines

### Commit Message Format

**Format**: `<type>: <short summary>`

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `test`: Adding or updating tests
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `chore`: Build process or auxiliary tool changes
- `style`: Code style changes (formatting, missing semi-colons, etc.)

**Examples**:
```
feat: add sentiment analysis tool for news headlines
fix: resolve async/sync mismatch in web search tool
docs: add configuration reference to README
test: add normalization tests for period formats
refactor: extract common tool validation logic to base class
perf: add TTL cache for real-time price data
chore: update langchain to 0.3.35
style: format code with black
```

### Commit Best Practices

- **Short summaries**: First line ≤50 characters
- **Imperative mood**: "add feature" not "added feature"
- **Present tense**: "change" not "changed"
- **Detailed body**: Explain why, not what (optional but recommended)

**Good Commit**:
```
feat: normalize LLM-generated tool arguments

The LLM sometimes emits expressions like period="3_years" or
latest=true that need normalization before passing to tools. This
adds normalization for period formats, boolean strings, and
symbol/ticker parameter names.

Fixes issue documented in GEMENINI.md:79
```

### Pull Request Guidelines

**Description Requirements**:
- Clear title summarizing the change
- Detailed description of what and why
- Rationale for the approach taken
- Scope of changes (files affected)
- Link to related issues or TODOs

**Additional Information**:
- Include CLI output or screenshots for UI/log changes
- Note provider impact (`yfinance` vs `financialdatasets`)
- Document breaking changes
- List any new dependencies

**PR Template**:
```markdown
## Summary
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- List of changes made
- Files affected

## Testing
- How was this tested?
- New tests added?

## Screenshots (if applicable)
CLI output or UI changes

## Provider Impact
- [ ] yfinance
- [ ] financialdatasets
- [ ] web

## Breaking Changes
None / List breaking changes

## Related Issues
Closes #123
See TODO.md task #5
```

### CI/CD Readiness

Before submitting PR:
- [ ] Ensure `uv sync` passes
- [ ] Local runs of `dexter-agent` succeed with sample queries
- [ ] All tests pass: `uv run pytest`
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No commented-out code or debug prints

---

## Security & Configuration Tips

### Environment Variables

**Setup**:
```bash
# Copy example to .env
cp env.example .env

# Edit .env with your API keys
# OPENAI_API_KEY=sk-...
# FINANCIAL_DATASETS_API_KEY=your-key-here
```

**Required**:
- `OPENAI_API_KEY`: Required for all configurations

**Optional**:
- `FINANCIAL_DATASETS_API_KEY`: Required when using `financialdatasets` provider
- `OPENAI_BASE_URL`: Override API endpoint (default: `https://api.openai.com/v1`)
- `OPENAI_MODEL`: Override model (default: `gpt-4`)
- `FUNCTION_MODEL`: Use separate model for tool selection
- `OPENAI_COMPAT_PROVIDER`: Specify provider type (`openai` or `ollama`)
- `DEXTER_DATA_PROVIDER`: Default data provider (`yfinance`, `financialdatasets`, or `web`)

### Provider Selection

**Default**: `yfinance` (free, no API key needed)

**Override Methods**:
1. Command-line flag: `--provider financialdatasets`
2. Environment variable: `DEXTER_DATA_PROVIDER=financialdatasets`

**Available Providers**:
- `yfinance`: Yahoo Finance (default, free)
- `financialdatasets`: Premium financial data API (requires subscription)
- `web`: DuckDuckGo web search only

### Security Best Practices

- Never commit `.env` files or API keys to git
- Use `.gitignore` to exclude sensitive files
- Keep tool calls idempotent and bounded
- Avoid committing large data files
- Rotate API keys periodically
- Use environment-specific `.env` files for dev/staging/prod

---

## Logging & LLM Output

### File Logging

**Enable**:
```bash
export LOGGING=true
```

**Configuration**:
- Default log file: `dexter.log`
- Override: `export DEXTER_LOG_FILE=custom.log`
- Format: JSON lines (one JSON object per line)

**What's Logged**:
- LLM request/response events only
- System prompts and user messages
- Structured outputs
- Token usage (if available)

**Not Logged**:
- Tool execution results (unless explicitly enabled)
- Internal debug information (use Python logging for that)

### Ollama Structured Output

When `OPENAI_BASE_URL` points to Ollama:
- Structured outputs use JSON mode
- Format specified via `extra_body={"format":"json"}`
- OpenAI-compatible endpoint required

**Example**:
```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=llama2
export OPENAI_COMPAT_PROVIDER=ollama
```

### Function Model

Use a separate model for tool selection and argument optimization:

```bash
export FUNCTION_MODEL=gpt-3.5-turbo
export FUNCTION_NUM_CTX=4096
```

**When to Use**:
- Cost optimization (use cheaper model for tool calls)
- Latency optimization (use faster model for tool calls)
- Specialization (use model fine-tuned for function calling)

### Prompt Truncation

Prevent context overflow:

```bash
export DEXTER_MAX_PROMPT_CHARS=10000
```

Hard-truncates system/user prompts to specified character limit.

### Parallel Tool Calls

Control concurrent tool execution:

```bash
export DEXTER_MAX_PARALLEL_TOOL_CALLS=8
```

Default: 4 concurrent executions

**Considerations**:
- Higher values = faster but more memory/API load
- Lower values = slower but more stable
- Recommended range: 2-10

### Empty Answer Fallback

If model returns empty or `"None"` for Answer schema:
- Automatic retry without structured output
- Falls back to plain text response
- Logged as warning

---

## Data Sources

Dexter supports the following data providers for financial information:

### Yahoo Finance (`yfinance`)

- **Status**: Default, primary data source
- **Cost**: Free
- **API Key**: Not required
- **Reliability**: Good for most use cases
- **Limitations**: Rate limits, occasionally incomplete data

**Available Data**:
- Stock prices (real-time and historical)
- Financial statements (income, balance sheet, cash flow)
- Financial metrics and ratios
- Analyst estimates
- Company news
- SEC filings (10-K, 10-Q, 8-K)

### Financial Datasets (`financialdatasets`)

- **Status**: Alternative, premium provider
- **Cost**: Paid subscription required
- **API Key**: Required (`FINANCIAL_DATASETS_API_KEY`)
- **Reliability**: Higher quality, more comprehensive
- **Limitations**: Cost, requires subscription

**Available Data**:
- Enhanced financial statements
- More comprehensive fundamental data
- Better data quality and consistency
- Faster API response times

**Base URL Override**:
```bash
export FINANCIAL_DATASETS_BASE_URL=https://custom-endpoint.com
```

### Web Search

- **Status**: Supplementary data source
- **Cost**: Free
- **Provider**: DuckDuckGo
- **Use Case**: Retrieving information not available via financial APIs

**Features**:
- RSS feed parsing utilities
- Abstract `BaseSearcher` class for pluggable search providers
- Configurable result limits

---

## Agent Capabilities (Tools)

The agent has access to a variety of tools to perform financial research.

### Financial Statement Tools

**Yahoo Finance**:
- `yf_get_income_statements(ticker, period)` - Income statements
- `yf_get_balance_sheets(ticker, period)` - Balance sheets
- `yf_get_cash_flow_statements(ticker, period)` - Cash flow statements

**Parameters**:
- `ticker`: Stock ticker symbol (e.g., "AAPL")
- `period`: "annual" or "quarterly"

### Price Data Tools

- `yf_get_price_snapshot(ticker)` - Current price and key metrics
- `yf_get_prices(ticker, period, interval)` - Historical prices

**Parameters**:
- `period`: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
- `interval`: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"

### Financial Metrics Tools

- `yf_get_financial_metrics_snapshot(ticker)` - Current metrics
- `yf_get_financial_metrics(ticker, period)` - Historical metrics

**Metrics Include**:
- P/E ratio, P/B ratio, P/S ratio
- ROE, ROA, profit margins
- Debt ratios, current ratio
- EPS, revenue per share

### News & Estimates

- `yf_get_news(ticker, limit)` - Latest news articles
- `yf_get_analyst_estimates(ticker)` - Analyst consensus estimates

### SEC Filings

- `yf_get_10K_filing_items(ticker)` - 10-K annual report items
- `yf_get_10Q_filing_items(ticker)` - 10-Q quarterly report items
- `yf_get_8K_filing_items(ticker)` - 8-K current report items
- `yf_get_filings(ticker, limit)` - List of all filings

### Web Search

- `web_search(query, max_results)` - DuckDuckGo search

**Use Cases**:
- Finding information not available in financial APIs
- Researching company products or strategy
- Finding recent news or announcements

---

## Additional Resources

- **README.md**: User-facing documentation and quick start guide
- **TODO.md**: Task tracking and improvement plans
- **pyproject.toml**: Dependencies and project configuration
- **env.example**: Example environment variables

---

*This document is maintained by the development team. Last updated: 2025-12-23*
