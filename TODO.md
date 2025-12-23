# Dexter TODO List

> Comprehensive task list for the Dexter financial research agent
>
> Status: In Progress
> Last Updated: 2025-12-23

---

## Quick Reference

- **Total Tasks**: 23
- **Critical**: 2 tasks
- **Important**: 3 tasks
- **Enhancements**: 7 tasks
- **Future Features**: 6 tasks
- **Architecture**: 3 tasks

---

## Table of Contents

1. [Critical Issues](#critical-issues) (Fix ASAP)
2. [Important Improvements](#important-improvements) (High Priority)
3. [Code Quality Enhancements](#code-quality-enhancements)
4. [Performance Optimizations](#performance-optimizations)
5. [Security Hardening](#security-hardening)
6. [Future Features & Extensions](#future-features--extensions)
7. [Architecture Enhancements](#architecture-enhancements) (New)
8. [Development Guidelines](#development-guidelines)
9. [Progress Tracking](#progress-tracking)

---

## Critical Issues

### 🔴 1. Fix Async/Sync Mismatch in Web Search Tool

**Priority**: Critical
**Status**: ❌ Not Started
**Effort**: 30 minutes
**Files**: `src/dexter/tools/web_search_tool.py`

**Problem**: The `web_search()` function is `async def` but LangChain tools are synchronous.

**Action Items**:
- [ ] Open `src/dexter/tools/web_search_tool.py`
- [ ] Check if `DuckDuckGoSearcher.get_search_results()` is actually async
- [ ] If async needed, wrap with `asyncio.run()`
- [ ] If not needed, remove `async`/`await` keywords
- [ ] Test with sample query
- [ ] Verify LangChain tool binding works

**Code Fix**:
```python
def web_search(query: str, max_results: int = 5) -> List[dict]:
    import asyncio
    searcher = DuckDuckGoSearcher()
    results = asyncio.run(searcher.get_search_results(query, max_results))
    return [result.dict() for result in results]
```

---

### 🔴 2. Replace Bare Exception Handlers

**Priority**: Critical
**Status**: ❌ Not Started
**Effort**: 1 hour
**Files**: `src/dexter/agent.py` (line 234), potentially others

**Problem**: Bare `except:` blocks catch SystemExit and KeyboardInterrupt, breaking Ctrl+C.

**Action Items**:
- [ ] Run `grep -rn "except:" src/dexter/ --include="*.py"` to find all occurrences
- [ ] Replace each with specific exception handling
- [ ] Add logging for caught exceptions
- [ ] Test that Ctrl+C still works

**Known Locations**:
- `src/dexter/agent.py:234` in `validate_task_completion()`

**Code Fix**:
```python
# Before
try:
    resp = call_llm(...)
    return resp.done
except:
    return False

# After
try:
    resp = call_llm(...)
    return resp.done
except Exception as e:
    logger.warning(f"Validation failed: {e}")
    return False
```

---

## Important Improvements

### 🟡 3. Add Comprehensive Test Coverage

**Priority**: Important
**Status**: ❌ Not Started
**Effort**: 2-3 days
**Current Coverage**: Only `tests/test_yf_shared.py` exists

**Action Items**:
- [ ] Add pytest dependencies: `pytest>=8.0.0`, `pytest-asyncio>=0.23.0`, `pytest-mock>=3.12.0`, `pytest-cov>=4.1.0`
- [ ] Create test structure:
  - [ ] `tests/conftest.py` - Shared fixtures
  - [ ] `tests/test_agent.py` - Agent orchestration logic
  - [ ] `tests/test_model.py` - LLM abstraction & fallbacks
  - [ ] `tests/test_tools.py` - Tool execution & validation
  - [ ] `tests/test_normalization.py` - Argument normalization
  - [ ] `tests/test_web_search.py` - Web search integration
  - [ ] `tests/integration/test_end_to_end.py` - Full queries
- [ ] Configure pytest in `pyproject.toml` with coverage reporting
- [ ] Add CI/CD integration (GitHub Actions)
- [ ] Target >70% code coverage

**Test Priorities**:
1. Task decomposition and planning
2. Tool selection logic
3. Loop detection mechanism
4. Parallel tool execution
5. LLM provider selection and fallbacks
6. Argument normalization edge cases

---

### 🟡 4. Centralize Configuration Management

**Priority**: Important
**Status**: ❌ Not Started
**Effort**: 2-3 hours
**Files**: `src/dexter/config.py`, `src/dexter/agent.py`, `src/dexter/model.py`

**Problem**: 12+ environment variables scattered throughout codebase without validation.

**Action Items**:
- [ ] Add `pydantic-settings>=2.0.0` to dependencies
- [ ] Rewrite `src/dexter/config.py` with complete Settings class
- [ ] Add field validators for all config options
- [ ] Create `validate_config()` function
- [ ] Update all imports from `os.getenv()` to `settings.*`
- [ ] Call `validate_config()` in CLI startup
- [ ] Document all env vars in README with table

**Configuration to Centralize**:
- API Keys: `OPENAI_API_KEY`, `FINANCIAL_DATASETS_API_KEY`
- LLM Config: `OPENAI_BASE_URL`, `OPENAI_MODEL`, `FUNCTION_MODEL`, `OPENAI_COMPAT_PROVIDER`
- Timeouts/Retries: `OPENAI_TIMEOUT`, `OPENAI_MAX_RETRIES`, `OPENAI_NUM_CTX`
- Agent Config: `DEXTER_MAX_STEPS`, `DEXTER_MAX_PARALLEL_TOOL_CALLS`, `DEXTER_MAX_PROMPT_CHARS`
- Data Provider: `DEXTER_DATA_PROVIDER`, `FINANCIAL_DATASETS_BASE_URL`
- Logging: `LOGGING`, `DEXTER_LOG_FILE`, `DEXTER_FORCE_PLAIN_RESPONSES`

---

### 🟡 5. Complete Argument Normalization

**Priority**: Important
**Status**: ❌ Not Started
**Effort**: 2-3 hours
**Files**: `src/dexter/agent.py`
**Documented In**: `GEMENINI.md:79`

**Problem**: LLM emits expressions like `period="3_years"` or `latest=true` that need normalization.

**Action Items**:
- [ ] Audit current `_normalize_tool_args()` method
- [ ] Add period expression normalization (`3_years` → `3y`, `quarterly` → `qtly`)
- [ ] Add boolean string normalization (`"true"` → `True`)
- [ ] Add symbol/ticker key normalization (`symbol` ↔ `ticker`)
- [ ] Add date format normalization (various → `YYYY-MM-DD`)
- [ ] Add logging for normalization events
- [ ] Create `tests/test_normalization.py` with comprehensive tests
- [ ] Update `GEMENINI.md` to mark as completed

**Normalization Examples**:
```python
# Period: "3_years" → "3y", "quarterly" → "qtly"
# Boolean: "true" → True, "false" → False
# Symbol: "symbol": "AAPL" → "ticker": "AAPL"
# Date: "12/31/2023" → "2023-12-31"
```



---

## Code Quality Enhancements

### 🟢 6. Improve Type Safety with mypy

**Priority**: Enhancement
**Status**: ❌ Not Started
**Effort**: 1 day
**Files**: All `.py` files

**Action Items**:
- [ ] Add mypy to dev dependencies: `mypy>=1.8.0`, `types-requests>=2.31.0`
- [ ] Create `mypy.ini` configuration
- [ ] Add type hints to functions missing them
- [ ] Fix type errors: `mypy src/dexter`
- [ ] Add mypy check to CI/CD pipeline
- [ ] Target zero mypy errors

**Focus Areas**:
- Return types for all public functions
- `List[BaseTool]` for tool selection methods
- Proper typing for LLM responses
- Optional types for nullable values



---

### 🟢 7. Enhance Error Logging with Structured Context

**Priority**: Enhancement
**Status**: ❌ Not Started
**Effort**: 2-3 hours
**Files**: `src/dexter/agent.py`, `src/dexter/model.py`, `src/dexter/tools/*.py`

**Action Items**:
- [ ] Update logging calls to include structured `extra={}` context
- [ ] Create `ExecutionContext` context manager
- [ ] Add JSON logging formatter option
- [ ] Ensure no sensitive data (API keys) in logs
- [ ] Test log parsing and analysis

**Structured Logging Example**:
```python
logger.error(
    "Tool execution failed",
    extra={
        "tool_name": tool_name,
        "args": tool_args,
        "task": current_task,
        "error": str(e),
        "error_type": type(e).__name__,
        "step": self.current_step,
    }
)
```

---

### 🟢 8. Reduce Code Duplication in Tools

**Priority**: Enhancement
**Status**: ❌ Not Started
**Effort**: 4-6 hours
**Files**: `src/dexter/tools/yf_*.py`, `src/dexter/tools/finance/*.py`

**Action Items**:
- [ ] Create `src/dexter/tools/base.py` with `BaseFinancialTool` abstract class
- [ ] Create `src/dexter/tools/decorators.py` with shared decorators
- [ ] Create `src/dexter/tools/formatters.py` for response formatting
- [ ] Refactor Yahoo Finance tools to use base classes
- [ ] Refactor FinancialDatasets tools to use base classes
- [ ] Write tests for base classes
- [ ] Target 50% reduction in duplicate code

**Common Patterns to Abstract**:
- Ticker validation and normalization
- Period parameter validation
- Error handling and logging
- Response formatting
- API rate limiting

---

### 🟢 9. Pin Dependency Versions More Strictly

**Priority**: Enhancement
**Status**: ❌ Not Started
**Effort**: 30 minutes
**Files**: `pyproject.toml`

**Action Items**:
- [ ] Update dependencies to pin major versions (e.g., `langchain>=0.3.27,<0.4.0`)
- [ ] Run `pip freeze > requirements-lock.txt`
- [ ] Document dependency update policy
- [ ] Consider switching to `poetry` or `pip-tools` for lock file management

**Dependencies to Pin**:
```toml
"langchain>=0.3.27,<0.4.0"
"langchain-openai>=0.3.35,<0.4.0"
"openai>=2.2.0,<3.0.0"
"pydantic>=2.11.10,<3.0.0"
```

---

### 🟢 10. Expand Documentation

**Priority**: Enhancement
**Status**: ❌ Not Started
**Effort**: 1 day
**Files**: Multiple

**Action Items**:
- [ ] Add docstrings to all private methods (methods starting with `_`)
- [ ] Create Mermaid architecture diagram in `GEMENINI.md`
- [ ] Add usage examples to key function docstrings
- [ ] Create `CONTRIBUTING.md` with:
  - Development setup instructions
  - Code style guidelines
  - Testing requirements
  - PR process
- [ ] Generate API documentation with Sphinx or MkDocs
- [ ] Add environment variable table to README.md
- [ ] Create example `.env` file

**Documentation Priorities**:
1. Architecture flow diagram (Planning → Action → Tool → Validation → Answer)
2. All public API methods documented
3. Configuration reference table
4. Contribution guidelines
5. Tool development guide

---

## Performance Optimizations

### 🟢 11. Add Caching Strategy

**Priority**: Optimization
**Status**: ❌ Not Started
**Effort**: 3-4 hours
**Files**: `src/dexter/agent.py`, `src/dexter/model.py`, `src/dexter/tools/*.py`

**Action Items**:
- [ ] Add `@lru_cache` for tool schema parsing
- [ ] Expand LLM instance caching (already partially implemented)
- [ ] Create `TTLCache` class for financial data with configurable TTLs
- [ ] Add cache configuration to Settings class
- [ ] Add cache metrics and logging (hit rate tracking)
- [ ] Write tests for caching behavior

**Cache TTL Recommendations**:
- Real-time prices: 60 seconds
- Fundamentals: 1 hour (3600 seconds)
- Filings: 24 hours (86400 seconds)
- Tool schemas: Cache indefinitely (LRU)
- LLM instances: Cache indefinitely (LRU)



---

### 🟢 12. Consider Async Tool Execution

**Priority**: Optimization (Future)
**Status**: ❌ Not Started
**Effort**: 1-2 days
**Files**: `src/dexter/agent.py`, `src/dexter/tools/*.py`

**Note**: Only implement if performance profiling shows I/O bottleneck.

**Action Items**:
- [ ] Evaluate feasibility and benefits
- [ ] Create async tool wrappers with `aiohttp`
- [ ] Convert I/O-bound tools to async
- [ ] Update agent execution with `asyncio.gather()`
- [ ] Maintain backward compatibility with sync wrapper
- [ ] Update CLI to support async execution
- [ ] Benchmark performance before/after
- [ ] Test for race conditions

**Evaluation Criteria**:
- Profile current tool execution times
- Identify I/O-bound operations
- Estimate improvement (should be >30% faster)
- Assess complexity vs. benefit tradeoff

---

## Security Hardening

### 🟢 13. Add API Key Validation at Startup

**Priority**: Security
**Status**: ❌ Not Started
**Effort**: 1 hour
**Files**: `src/dexter/config.py`, `src/dexter/cli.py`

**Action Items**:
- [ ] Enhance `validate_config()` with comprehensive checks
- [ ] Add API key format validation (regex patterns)
- [ ] Add provider-specific validation (e.g., FinancialDatasets requires its key)
- [ ] Add numeric range validation (max_steps > 0, etc.)
- [ ] Call validation in CLI startup with clear error messages
- [ ] Optional: Test API connectivity on startup
- [ ] Ensure no API keys logged in error messages

**Validation Checks**:
- `OPENAI_API_KEY` required and matches format `sk-[A-Za-z0-9]{48}`
- `FINANCIAL_DATASETS_API_KEY` required when provider is `financialdatasets`
- `max_steps` > 0
- `max_parallel_tool_calls` between 1 and 20
- `data_provider` in allowed list

---

### 🟢 14. Add Input Sanitization for User Queries

**Priority**: Security
**Status**: ❌ Not Started
**Effort**: 1-2 hours
**Files**: `src/dexter/cli.py`, `src/dexter/agent.py`

**Action Items**:
- [ ] Create `src/dexter/utils/security.py` with `InputSanitizer` class
- [ ] Implement query sanitization (length limit, dangerous patterns)
- [ ] Implement ticker sanitization (uppercase, alphanumeric only)
- [ ] Apply sanitization in CLI input loop
- [ ] Apply sanitization in tool argument normalization
- [ ] Optional: Add rate limiting for queries
- [ ] Create `tests/test_security.py` with edge cases

**Dangerous Patterns to Block**:
- Command injection: `; rm -rf`, `; shutdown`
- Shell execution: backticks, `$(...)`, pipes
- Null bytes: `\x00`
- Excessive length: > 5000 characters

**Sanitization Rules**:
```python
# Query: Strip, limit length, remove null bytes, check patterns
# Ticker: Strip, uppercase, limit to 10 chars, alphanumeric + dots/hyphens only
```

---

## Future Features & Extensions

### 🌟 15. Add New Data Sources

**Priority**: Feature
**Status**: ❌ Not Started
**Effort**: Variable (per provider)
**Source**: `GEMENINI.md`

**Potential Providers**:
- [ ] Alpha Vantage API
- [ ] Polygon.io API
- [ ] IEX Cloud API
- [ ] Quandl/Nasdaq Data Link
- [ ] Federal Reserve Economic Data (FRED)

**Implementation Steps**:
1. Create new tool files in `src/dexter/tools/[provider]/`
2. Implement tools following existing patterns
3. Register provider in `src/dexter/tools/__init__.py`
4. Add provider to Settings validation
5. Document new provider in README
6. Add tests for new tools

## Architecture Enhancements

### 🏗️ 21. Implement Semantic Tool Selection

**Priority**: Important
**Status**: ❌ Not Started
**Effort**: 3-4 hours
**Files**: `src/dexter/agent.py`

**Problem**: Current tool selection uses string matching which can be brittle and limited.

**Action Items**:
- [ ] Research embedding-based tool selection approaches
- [ ] Create tool embedding database using sentence-transformers
- [ ] Implement semantic similarity matching in `_select_tools_for_task()`
- [ ] Add fallback to current string matching for compatibility
- [ ] Create `tests/test_tool_selection.py` with semantic test cases
- [ ] Benchmark performance impact

**Implementation Approach**:
```python
# Use sentence embeddings for semantic matching
from sentence_transformers import SentenceTransformer

class ToolSelector:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tool_embeddings = {}
        
    def add_tool(self, tool_name, description):
        self.tool_embeddings[tool_name] = self.model.encode(description)
        
    def select_tools(self, task_description, threshold=0.7):
        task_embedding = self.model.encode(task_description)
        # Return tools with similarity > threshold
```

### 🏗️ 22. Implement Plugin Architecture

**Priority**: Enhancement
**Status**: ❌ Not Started
**Effort**: 2-3 days
**Files**: `src/dexter/tools/__init__.py`, `src/dexter/agent.py`

**Problem**: Adding new tools and providers requires code changes in multiple files.

**Action Items**:
- [ ] Research Python plugin architectures (entry points, importlib)
- [ ] Create plugin interface for tools and data providers
- [ ] Implement dynamic plugin discovery and loading
- [ ] Create plugin documentation and examples
- [ ] Update tool registration system to support plugins
- [ ] Create sample plugin for demonstration

**Implementation Approach**:
```python
# Plugin interface
class DexterPlugin(ABC):
    @abstractmethod
    def get_tools(self) -> List[Callable]:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

# Plugin discovery
class PluginManager:
    def __init__(self):
        self.plugins = {}
        
    def discover_plugins(self):
        # Use entry_points or file scanning
        pass
        
    def load_plugin(self, plugin_class):
        plugin = plugin_class()
        self.plugins[plugin.get_name()] = plugin
```

### 🏗️ 23. Enhance State Management

**Priority**: Optimization
**Status**: ❌ Not Started
**Effort**: 2-3 days
**Files**: `src/dexter/agent.py`, `src/dexter/cli.py`

**Problem**: Current state management is limited to single query execution.

**Action Items**:
- [ ] Research conversation state management patterns
- [ ] Implement session persistence for multi-turn conversations
- [ ] Add conversation history tracking
- [ ] Implement state serialization/deserialization
- [ ] Add session management commands to CLI
- [ ] Create `tests/test_state_management.py`

**Implementation Approach**:
```python
class ConversationState:
    def __init__(self):
        self.history = []
        self.current_session_id = str(uuid4())
        self.session_data = {}
        
    def add_to_history(self, query, answer, tools_used):
        self.history.append({
            'timestamp': datetime.now(),
            'query': query,
            'answer': answer,
            'tools_used': tools_used
        })
        
    def save_state(self, file_path):
        # Serialize to JSON
        pass
        
    def load_state(self, file_path):
        # Deserialize from JSON
        pass
```

---

### 🌟 16. Add Analysis Tools

**Priority**: Feature
**Status**: ❌ Not Started
**Effort**: 1-2 weeks
**Source**: `GEMENINI.md`

**Sentiment Analysis**:
- [ ] Create tool to analyze news headlines
- [ ] Return sentiment score (-1 to +1)
- [ ] Use pre-trained models (VADER, FinBERT)
- [ ] Cache sentiment results with TTL

**Technical Analysis**:
- [ ] Moving averages (SMA, EMA)
- [ ] Momentum indicators (RSI, MACD)
- [ ] Volatility indicators (Bollinger Bands)
- [ ] Volume indicators (OBV)
- [ ] Chart pattern recognition (optional)

**Implementation**:
- Create `src/dexter/tools/analysis/` directory
- Use `ta-lib` or `pandas-ta` library
- Return structured indicator values
- Add visualization option (matplotlib/plotly)

---

### 🌟 17. Improve Output Formatting

**Priority**: Feature
**Status**: ❌ Not Started
**Effort**: 1 week
**Source**: `GEMENINI.md`

**Action Items**:
- [ ] Modify `_generate_answer()` in `src/dexter/agent.py`
- [ ] Add Markdown table formatting for structured data
- [ ] Add JSON output mode (`--format json`)
- [ ] Add HTML report generation (`--format html`)
- [ ] Add chart/graph embedding (optional)
- [ ] Make output format configurable

**Output Formats**:
1. **Markdown** (default): Rich text with tables
2. **JSON**: Structured data for programmatic use
3. **HTML**: Interactive reports with charts
4. **Plain text**: Simple, no formatting

---

### 🌟 18. Implement Web Scraping

**Priority**: Feature
**Status**: ❌ Not Started
**Effort**: 1-2 weeks
**Source**: `GEMENINI.md`

**Target Sites**:
- [ ] SEC EDGAR (filings, insider trades)
- [ ] Company investor relations pages
- [ ] Financial news sites (WSJ, Bloomberg, etc.)
- [ ] Earnings call transcripts
- [ ] Economic calendars

**Implementation**:
- [ ] Use BeautifulSoup or Scrapy
- [ ] Add rate limiting and robots.txt compliance
- [ ] Implement retry logic with exponential backoff
- [ ] Cache scraped data with TTL
- [ ] Add error handling for site changes
- [ ] Respect terms of service

**Tools to Create**:
- `scrape_company_website(ticker, url)`
- `scrape_sec_filing(ticker, filing_type, date)`
- `scrape_earnings_transcript(ticker, quarter, year)`

---

### 🌟 19. Persist Answers & Enhanced Caching

**Priority**: Feature
**Status**: ❌ Not Started
**Effort**: 1 week
**Source**: `GEMENINI.md`

**Answer Persistence**:
- [ ] Store final answers in SQLite or PostgreSQL
- [ ] Include timestamp, query, answer, tools used
- [ ] Track answer versions for same query over time
- [ ] Add `--history` command to view past queries
- [ ] Add trend analysis across historical answers

**Tool Output Caching**:
- [ ] Create caching layer: `(provider, tool, params)` → result
- [ ] Use TTL based on data freshness requirements
- [ ] Implement cache invalidation on demand
- [ ] Add cache statistics dashboard
- [ ] Support Redis for distributed caching (optional)

**Database Schema**:
```sql
CREATE TABLE queries (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    query TEXT,
    answer TEXT,
    tools_used JSON,
    execution_time_ms INTEGER
);

CREATE TABLE tool_cache (
    cache_key TEXT PRIMARY KEY,
    result JSON,
    cached_at DATETIME,
    ttl_seconds INTEGER
);
```

---

### 🌟 20. Add Multi-Modal Support

**Priority**: Future
**Status**: ❌ Not Started
**Effort**: 2-3 weeks

**Chart Generation**:
- [ ] Generate price charts with matplotlib/plotly
- [ ] Generate financial statement visualizations
- [ ] Generate comparison charts (multiple tickers)
- [ ] Export charts as PNG/SVG

**Image Analysis**:
- [ ] Analyze uploaded chart images
- [ ] Extract data from financial infographics
- [ ] OCR for scanned documents
- [ ] Use GPT-4 Vision or similar models

**PDF Processing**:
- [ ] Parse 10-K/10-Q PDFs directly
- [ ] Extract tables from PDF filings
- [ ] Summarize lengthy documents
- [ ] Index PDFs for semantic search

---

## Development Guidelines

### Project Structure

```
src/dexter/
├── agent.py              # Main orchestration (Agent class)
├── cli.py                # CLI entry point
├── model.py              # LLM abstraction layer
├── prompts.py            # System prompts
├── schemas.py            # Pydantic models
├── config.py             # Configuration management
├── tools/
│   ├── __init__.py       # Tool factory
│   ├── yf_*.py           # Yahoo Finance tools
│   ├── finance/          # FinancialDatasets tools
│   ├── web_search/       # Web search tools
│   └── base.py           # Base classes (to be created)
└── utils/
    ├── logger.py         # Logging utilities
    ├── ui.py             # Terminal UI
    └── security.py       # Input sanitization (to be created)
```

### Coding Standards

**Style**:
- Follow PEP 8
- 4-space indentation
- Max line length: 100 characters
- Use type hints for all functions
- Docstrings for public methods (Google style)

**Naming**:
- Modules: `lowercase_underscores.py`
- Functions/Variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Private methods: `_leading_underscore`

**Organization**:
- User-facing commands: `cli.py`
- Data access: `tools/`
- Shared utilities: `utils/`
- Data models: `schemas.py`
- Business logic: `agent.py`

### Testing Standards

**Framework**: pytest

**Structure**:
```
tests/
├── conftest.py              # Shared fixtures
├── test_agent.py
├── test_model.py
├── test_tools.py
├── test_normalization.py
├── test_security.py
└── integration/
    └── test_end_to_end.py
```

**Requirements**:
- All new features must include tests
- Maintain >70% code coverage
- Use mocks for external API calls
- Test edge cases and error conditions
- Run `uv run pytest` before committing

### Development Commands

```bash
# Install dependencies
uv sync

# Run agent locally
uv run dexter-agent [--provider yfinance|financialdatasets|web]

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/dexter --cov-report=html

# Type checking (after mypy setup)
mypy src/dexter

# Format code (if black/ruff configured)
black src/dexter tests/
```

### Commit Guidelines

**Format**: `<type>: <short summary>`

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build/tooling changes

**Examples**:
```
feat: add sentiment analysis tool
fix: resolve async/sync mismatch in web search
docs: add configuration reference to README
test: add normalization tests for period formats
refactor: extract common tool validation logic
perf: add TTL cache for price data
```

---

## Progress Tracking

### Summary

| Category | Completed | Total | Progress |
|----------|-----------|-------|----------|
| 🔴 Critical Issues | 0 | 2 | 0% |
| 🟡 Important | 0 | 3 | 0% |
| 🟢 Code Quality | 0 | 5 | 0% |
| 🟢 Performance | 0 | 2 | 0% |
| 🟢 Security | 0 | 2 | 0% |
| 🌟 Future Features | 0 | 6 | 0% |
| 🏗️ Architecture | 0 | 3 | 0% |
| **Total** | **0** | **23** | **0%** |

### Implementation Phases

**Phase 1: Critical Fixes** (Week 1)
- [ ] Fix async/sync mismatch (#1)
- [ ] Replace bare exception handlers (#2)

**Phase 2: Core Improvements** (Weeks 2-3)
- [ ] Add test coverage (#3)
- [ ] Centralize configuration (#4)
- [ ] Complete argument normalization (#5)

**Phase 3: Code Quality** (Weeks 4-5)
- [ ] Type safety with mypy (#6)
- [ ] Enhanced logging (#7)
- [ ] Reduce code duplication (#8)
- [ ] Pin dependencies (#9)
- [ ] Expand documentation (#10)

**Phase 4: Performance & Security** (Week 6)
- [ ] Add caching strategy (#11)
- [ ] API key validation (#13)
- [ ] Input sanitization (#14)

**Phase 5: Architecture Enhancements** (Weeks 7-8)
- [ ] Implement semantic tool selection (#21)
- [ ] Implement plugin architecture (#22)
- [ ] Enhance state management (#23)

**Phase 6: Advanced Features** (Weeks 9+)
- [ ] Async tool execution (if needed) (#12)
- [ ] New data sources (#15)
- [ ] Analysis tools (#16)
- [ ] Output formatting (#17)
- [ ] Web scraping (#18)
- [ ] Answer persistence (#19)
- [ ] Multi-modal support (#20)

### Next Actions

**Immediate** (This Week):
1. Fix async/sync mismatch in web search tool
2. Replace bare exception handlers
3. Start test infrastructure setup

**Short-term** (This Month):
1. Complete test coverage for core components
2. Centralize configuration management
3. Implement argument normalization improvements

**Long-term** (Next Quarter):
1. Add comprehensive documentation
2. Implement caching and performance optimizations
3. Begin work on new analysis tools

---

## Notes

- **Living Document**: Update this TODO as tasks are completed or priorities change
- **Parallel Work**: Many tasks can be worked on independently (e.g., tests + config)
- **Incremental**: Commit often, test after each change
- **Track Issues**: Consider creating GitHub issues for each major task
- **Code Review**: Review PRs after each phase completion
- **Priorities**: Adjust based on immediate needs and user feedback

---

## Getting Started

**To start working on these tasks**:

1. **Choose a task**: Start with Critical Issues (#1-2)
2. **Create a branch**: `git checkout -b fix/async-web-search` or `git checkout -b improvement/add-tests`
3. **Make changes**: Follow the action items and code examples
4. **Test thoroughly**: Run tests and manual verification
5. **Commit**: Use conventional commit format
6. **Update this file**: Check off completed items
7. **Open PR**: Include clear description and link to this TODO

**Questions or blockers?** Document them here or create an issue.

---

*Last Updated: 2025-12-23 | Total Tasks: 23 | Completed: 0*
