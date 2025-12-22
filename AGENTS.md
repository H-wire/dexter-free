# Repository Guidelines

## Project Structure & Module Organization
- `src/dexter/`: Python package entry. Key modules: `agent.py` (orchestration), `cli.py` (CLI), `model.py` (LLM calls), `prompts.py`, `schemas.py`, `utils/` (UI, logging, intro), and `tools/`.
- `src/dexter/tools/finance/`: FinancialDatasets-backed tools; `prices.py`, `fundamentals.py`, `metrics.py`, etc.
- `src/dexter/tools/yf_*`: Yahoo Finance tools (default provider).
- Top-level: `pyproject.toml` (Python build), `uv.lock`, `index.js` (Node wrapper), `package.json`, `env.example`, `README.md`.

## Build, Test, and Development Commands
- `uv sync`: Install Python dependencies from `pyproject.toml`/`uv.lock`.
- `uv run dexter-agent [--provider yfinance|financialdatasets]`: Run the CLI locally.
- `node index.js --provider yfinance`: Invoke via the Node wrapper (mirrors `dexter-agent`).
- `uv run python -m dexter.cli --provider yfinance`: Alternate entry for debugging.
- Tests: none committed yet; see Testing Guidelines to add `pytest`.

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indent, type hints required; docstrings for public functions/classes.
- Naming: modules `lowercase_underscores`, functions/vars `snake_case`, classes `PascalCase`.
- Organization: place user-facing commands in `cli.py`; pure data access in `tools/`; shared helpers in `utils/`; data models in `schemas.py`.
- Logging/UI: use `dexter.utils.logger.Logger` and `dexter.utils.ui.show_progress` for consistent output.

## Testing Guidelines
- Framework: prefer `pytest`. Put tests under `tests/` mirroring `src/dexter` (e.g., `tests/tools/test_yf_prices.py`).
- Naming: files `test_*.py`; functions `test_*`.
- Run: `uv run pytest -q`.
- Add minimal fixtures/mocks for networked tools; do not hit external APIs in unit tests.

## Commit & Pull Request Guidelines
- Commits: short, imperative summaries (e.g., `default to yfinance provider`, `normalize yfinance news parsing`).
- PRs: clear description, rationale, and scope; link issues; include CLI output or screenshots for UI/log changes; note provider impact (`yfinance` vs `financialdatasets`).
- CI/readiness: ensure `uv sync` passes and local runs of `dexter-agent` succeed with sample queries.

## Security & Configuration Tips
- Env: copy `env.example` to `.env`; set `OPENAI_API_KEY`; optionally `FINANCIAL_DATASETS_API_KEY`.
- Provider: default is `yfinance`; override with `--provider` or `DEXTER_DATA_PROVIDER`.
- Avoid committing secrets or large data; keep tool calls idempotent and bounded.

## Logging & LLM Output (Local/Ollama)
- File logging: set `LOGGING=true` to enable file logging; default log file is `dexter.log` (override with `DEXTER_LOG_FILE`).
- Log contents: only LLM request/response events are written as JSON lines (one JSON object per line).
- Ollama structured output: when `OPENAI_BASE_URL` points to Ollama, structured outputs use JSON mode via `extra_body={"format":"json"}` (OpenAI-compatible endpoint).
- Function model: set `FUNCTION_MODEL` to use a separate model for tool selection/argument optimization; `FUNCTION_NUM_CTX` controls its context window.
- Prompt truncation: `DEXTER_MAX_PROMPT_CHARS` hard-truncates system/user prompts to avoid overrunning context.
- Parallel tool calls: `DEXTER_MAX_PARALLEL_TOOL_CALLS` controls concurrent tool execution.
- Empty answer fallback: if the model returns empty or `"None"` for the Answer schema, Dexter retries once without structured output.

## TODOs
- Local OpenAI-compatible API support: Add env-driven LLM config in `model.py` with `get_llm()` that reads `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY` (optional), `OPENAI_TIMEOUT`, `OPENAI_MAX_RETRIES`. Use `ChatOpenAI(base_url=..., model=..., api_key=...)`; if `OPENAI_BASE_URL` is set but no key, default to `sk-local`. Optionally add CLI flags `--openai-base-url`/`--openai-model`. Update `env.example` and README with examples (e.g., `http://localhost:1234/v1`, Ollama/LM Studio). Ensure graceful fallback when structured output/tool-calls are unsupported.
