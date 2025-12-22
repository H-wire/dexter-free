# Dexter Architecture Overview

This document outlines the architecture of Dexter, a financial research agent powered by Large Language Models (LLMs).

## High-Level Summary

Dexter is a sophisticated financial research agent built on the LangChain framework. Its architecture is designed around a structured, multi-step reasoning process orchestrated by the `Agent` class in `src/dexter/agent.py`. The agent breaks down a user's query into a series of tasks, executes them, and then synthesizes an answer. This multi-step process, combined with self-correction and dynamic tool use, makes for a robust and powerful research assistant.

## Core Workflow

The application's core logic resides in the `Agent.run` method within `src/dexter/agent.py`. The workflow is as follows:

1.  **Planning:** The user's query is first processed by an LLM using a specialized `PLANNING_SYSTEM_PROMPT` (from `src/dexter/prompts.py`). The goal is to break the query down into a series of discrete, actionable tasks. The output is structured as a `TaskList` schema (from `src/dexter/schemas.py`).

2.  **Execution Loop:** The agent iterates through each task identified in the planning phase. For each task, it enters a sub-loop to perform the necessary actions.

3.  **Action:** Inside the task loop, the agent consults the LLM again with the `ACTION_SYSTEM_PROMPT`. It provides the LLM with a set of available tools and asks it to choose the next action to perform to accomplish the current task.

4.  **Argument Optimization:** In a unique and crucial step, if the LLM decides to use a tool, the agent makes *another* LLM call. This time, it uses the `TOOL_ARGS_SYSTEM_PROMPT` to refine and optimize the parameters for the chosen tool. This ensures the tool is called with the most effective arguments.

5.  **Tool Execution:** The agent executes the specified tool with the newly optimized arguments. The tool system is designed to be pluggable.

6.  **Validation:** After executing the tool, the agent performs a validation step. It asks the LLM if the current task is complete, using the `VALIDATION_SYSTEM_PROMPT` and expecting an `IsDone` structured response. If the task is not yet complete, the action loop continues.

7.  **Answer Synthesis:** Once all tasks in the main plan are marked as complete, the agent aggregates all the outputs and observations gathered from the tool executions. It sends this collected evidence to the LLM one final time with the `ANSWER_SYSTEM_PROMPT` to generate a comprehensive, user-facing answer.

## Key Components

-   **`src/dexter/agent.py`**: The central orchestrator. The `Agent` class manages the entire workflow, from planning to execution and final response generation.

-   **`src/dexter/model.py`**: This file provides a critical abstraction layer for all LLM interactions. The `call_llm` function handles model configuration, structured output binding (ensuring the LLM's response conforms to a Pydantic schema), tool binding, and error handling. It's designed for compatibility with various models, including local Ollama instances.

-   **`src/dexter/prompts.py`**: This file can be considered the "brain" of the agent. It contains all the system prompts that guide the LLM's reasoning at each specific step of the workflow. The clarity and detail of these prompts are fundamental to the agent's performance.

-   **`src/dexter/schemas.py`**: Defines the Pydantic models (`TaskList`, `IsDone`, `Answer`, etc.) used to enforce a strict, predictable structure on the LLM's outputs. This is essential for making the agent's behavior reliable.

-   **`src/dexter/tools/__init__.py`**: This file implements the pluggable tool architecture. It acts as a factory, providing the correct set of tools based on the selected data provider (e.g., `yfinance` or `financialdatasets`). This separation of concerns makes it easy to extend the agent with new data sources.

-   **`src/dexter/cli.py`**: The main entry point for the application. It handles command-line arguments, environment setup, and the instantiation and startup of the main `Agent` loop.

## Data Sources

Dexter supports the following data providers for financial information:

-   **Yahoo Finance (`yfinance`)**: This is the default and primary data source for various financial tools, including prices, financials, news, and estimates.
-   **Financial Datasets (`financialdatasets`)**: An alternative data provider. While the specific implementation details for this provider are not fully detailed, it is designed to offer similar financial data capabilities.

## Agent Capabilities (Tools)

The agent has access to a variety of tools to perform financial research.

-   **Web Search**: The system includes a generic web search capability, primarily through an abstract `BaseSearcher` class that provides utilities for parsing RSS feeds. It offers a framework for integrating different search providers, but does not explicitly use a service like Tavily.
-   **Analyst Estimates**: Fetches analyst estimates for a given stock ticker from the configured financial data provider.
-   **Company Filings**: Retrieves company filings such as 10-K, 10-Q, and 8-K reports from the configured financial data provider.
-   **Financial Statements**: Fetches comprehensive financial statements, including:
    -   Income Statements
    -   Balance Sheets
    -   Cash Flow Statements
-   **Key Metrics**: Gathers key financial metrics and ratios for a company.
-   **News**: Fetches the latest news articles for a specific stock or company.
-   **Price History**: Retrieves historical stock price data for a given ticker.

## Future Plans & Extension Points

Here are a few ways the agent's capabilities could be enriched in the future:

1.  **Add New Data Sources**: Integrate new financial data APIs (e.g., Alpha Vantage, Polygon.io) by creating new tool files in the `src/dexter/tools/` directory and registering them in `src/dexter/tools/__init__.py`.

2.  **Add New Analysis Tools**: Create tools that perform new types of analysis on the retrieved data. For example:
    *   **Sentiment Analysis**: A tool that analyzes news headlines and returns a sentiment score.
    *   **Technical Analysis**: A tool to calculate technical indicators like moving averages or RSI from price data.

3.  **Improve Output Formatting**: Modify the `_generate_answer` method in `src/dexter/agent.py` to customize the final report, such as generating Markdown tables or a structured JSON output.

4.  **Implement Web Scraping**: Use libraries like BeautifulSoup or Scrapy to create tools that can scrape specific websites for information not available through APIs.

5.  **Persist Answers & Cache Tool Outputs**: Store final answers with timestamps for trend tracking over time, and add a tool-output cache (provider + tool + params with TTL) to avoid refetching unchanged data.
