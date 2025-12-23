import json
import os
import re
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from langchain_core.messages import AIMessage

from dexter.model import call_llm
from dexter.prompts import (
    ACTION_SYSTEM_PROMPT,
    ACTION_JSON_SYSTEM_PROMPT,
    get_answer_system_prompt,
    PLANNING_SYSTEM_PROMPT,
    get_tool_args_system_prompt,
    VALIDATION_SYSTEM_PROMPT,
)
from dexter.schemas import Answer, IsDone, OptimizedToolArgs, Task, TaskList, ToolCallList
from dexter.tools import AVAILABLE_DATA_PROVIDERS, get_tools
from dexter.utils.logger import Logger
from dexter.utils.ui import show_progress


class Agent:
    def __init__(
        self,
        max_steps: int = 20,
        max_steps_per_task: int = 5,
        data_provider: str = "yfinance",
    ):
        self.logger = Logger()
        self.max_steps = max_steps            # global safety cap
        self.max_steps_per_task = max_steps_per_task
        try:
            self.max_parallel_tool_calls = int(os.getenv("DEXTER_MAX_PARALLEL_TOOL_CALLS", "4"))
        except ValueError:
            self.max_parallel_tool_calls = 4
        provider_key = data_provider.lower()
        if provider_key not in AVAILABLE_DATA_PROVIDERS:
            self.logger._log(
                f"Unknown data provider '{data_provider}'. Falling back to Yahoo Finance."
            )
            provider_key = "yfinance"
        self.data_provider = provider_key
        self.tools = get_tools(provider_key)
        self.logger._log(f"Agent initialized with data provider: {self.data_provider}")
        disable_flag = os.getenv("DEXTER_DISABLE_TOOL_ARG_OPTIMIZATION", "").strip().lower()
        self.disable_tool_arg_optimization = disable_flag in {"1", "true", "yes", "on"}

    def _tool_descriptions(self) -> str:
        return "\n".join([f"- {t.name}: {t.description}" for t in self.tools])

    def _coerce_tool_calls(self, tool_call_list: ToolCallList, allowed_tools: List) -> list[dict]:
        valid_names = {t.name for t in allowed_tools}
        coerced = []
        for call in tool_call_list.tool_calls:
            if call.name not in valid_names:
                continue
            coerced.append(
                {
                    "name": call.name,
                    "args": call.args,
                    "id": f"call_{uuid4().hex}",
                    "type": "tool_call",
                }
            )
        return coerced

    def _select_tools_for_task(self, task_desc: str) -> List:
        desc = task_desc.lower()
        selected = []

        def include_if_name_contains(substrs: list[str]) -> None:
            for tool in self.tools:
                name = tool.name.lower()
                if any(s in name for s in substrs):
                    selected.append(tool)

        if any(k in desc for k in ["10-q", "10q", "quarterly report", "quarterly filing"]):
            include_if_name_contains(["10q", "filings"])
        if any(k in desc for k in ["10-k", "10k", "annual report", "annual filing"]):
            include_if_name_contains(["10k", "filings"])
        if any(k in desc for k in ["8-k", "8k"]):
            include_if_name_contains(["8k", "filings"])
        if "filing" in desc:
            include_if_name_contains(["filings"])
        if "income statement" in desc or "income" in desc:
            include_if_name_contains(["income_statements"])
        if "balance sheet" in desc:
            include_if_name_contains(["balance_sheets"])
        if "cash flow" in desc:
            include_if_name_contains(["cash_flow"])
        if "price" in desc or "ticker" in desc or "symbol" in desc:
            include_if_name_contains(["price_snapshot", "prices"])
        if "news" in desc:
            include_if_name_contains(["news"])
        if "analyst" in desc or "estimates" in desc:
            include_if_name_contains(["analyst_estimates", "estimates"])
        if "metric" in desc:
            include_if_name_contains(["metrics"])

        # De-duplicate while preserving order.
        seen = set()
        unique = []
        for tool in selected:
            if tool.name not in seen:
                unique.append(tool)
                seen.add(tool.name)
        return unique if unique else list(self.tools)

    def _extract_json_from_text(self, text: str) -> Optional[dict]:
        """Try to find the first JSON object in a plain-text message."""
        if not text:
            return None
        matches = re.findall(r"\{.*?\}", text, re.DOTALL)
        for candidate in matches:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    def _parse_tasks_from_text(self, text: str) -> list[Task]:
        """Parse TaskList JSON from an LLM text response when structured output is disabled."""
        payload = self._extract_json_from_text(text)
        if not payload or not isinstance(payload, dict):
            return []
        tasks_data = payload.get("tasks")
        if not isinstance(tasks_data, list):
            return []
        parsed_tasks: list[Task] = []
        for entry in tasks_data:
            if isinstance(entry, Task):
                parsed_tasks.append(entry)
            elif isinstance(entry, dict):
                try:
                    parsed_tasks.append(Task.model_validate(entry))
                except Exception:
                    continue
        return parsed_tasks

    # ---------- task planning ----------
    @show_progress("Planning tasks...", "Tasks planned")
    def plan_tasks(self, query: str) -> List[Task]:
        prompt = f"""
        Given the user query: "{query}",
        Create a list of tasks to be completed.
        Example: {{"tasks": [{{"id": 1, "description": "some task", "done": false}}]}}
        """
        system_prompt = PLANNING_SYSTEM_PROMPT.format(tools=self._tool_descriptions())
        try:
            response = call_llm(prompt, system_prompt=system_prompt, output_schema=TaskList)
            if isinstance(response, TaskList):
                tasks = response.tasks
            elif isinstance(response, dict):
                tasks = TaskList.model_validate(response).tasks
            else:
                raw_text = getattr(response, "content", "") if hasattr(response, "content") else str(response)
                tasks = self._parse_tasks_from_text(raw_text)
        except Exception as e:
            # Try to salvage tasks if LLM returned a plain message
            fallback_message = getattr(e, "response", None)
            if isinstance(fallback_message, AIMessage):
                tasks = self._parse_tasks_from_text(getattr(fallback_message, "content", ""))
            else:
                self.logger._log(f"Planning failed: {e}")
                tasks = []
        if not tasks:
            tasks = [Task(id=1, description=query, done=False)]
        
        task_dicts = [task.dict() for task in tasks]
        self.logger.log_task_list(task_dicts)
        return tasks

    # ---------- ask LLM what to do ----------
    @show_progress("Thinking...", "")
    def ask_for_actions(self, task_desc: str, last_outputs: str = "") -> AIMessage:
        # last_outputs = textual feedback of what we just tried
        prompt = f"""
        We are working on: "{task_desc}".
        Here is a history of tool outputs from the session so far: {last_outputs}

        Based on the task and the outputs, what should be the next step?
        """
        try:
            function_model = os.getenv("FUNCTION_MODEL")
            if not function_model:
                return call_llm(
                    prompt,
                    system_prompt=ACTION_SYSTEM_PROMPT,
                    tools=self.tools,
                )
            selected_tools = self._select_tools_for_task(task_desc)
            tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in selected_tools])
            json_prompt = f"""
            Available tools (name: description):
            {tool_descriptions}

            Task: "{task_desc}"
            History: {last_outputs}
            """
            json_response = call_llm(
                json_prompt,
                system_prompt=ACTION_JSON_SYSTEM_PROMPT,
                output_schema=ToolCallList,
                model_override=function_model,
            )
            if isinstance(json_response, dict):
                json_response = ToolCallList.model_validate(json_response)
            if isinstance(json_response, ToolCallList):
                tool_calls = self._coerce_tool_calls(json_response, selected_tools)
                if tool_calls:
                    self.logger._log("Function model returned JSON tool calls.")
                    return AIMessage(content="", tool_calls=tool_calls)
            self.logger._log("Function model returned no tool calls.")
            return AIMessage(content="", tool_calls=[])
        except Exception as e:
            self.logger._log(f"ask_for_actions failed: {e}")
            return AIMessage(content="Failed to get actions.")

    # ---------- ask LLM if task is done ----------
    @show_progress("Validating...", "")
    def ask_if_done(self, task_desc: str, recent_results: str) -> bool:
        prompt = f"""
        We were trying to complete the task: "{task_desc}".
        Here is a history of tool outputs from the session so far: {recent_results}

        Is the task done?
        """
        try:
            resp = call_llm(prompt, system_prompt=VALIDATION_SYSTEM_PROMPT, output_schema=IsDone)
            return resp.done
        except:
            return False

    # ---------- optimize tool arguments ----------
    @show_progress("Optimizing tool call...", "")
    def optimize_tool_args(self, tool_name: str, initial_args: dict, task_desc: str) -> dict:
        """Optimize tool arguments based on task requirements."""
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            return initial_args
        
        # Get tool schema info
        tool_description = tool.description
        tool_schema = tool.args_schema.schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}
        
        prompt = f"""
        Task: "{task_desc}"
        Tool: {tool_name}
        Tool Description: {tool_description}
        Tool Parameters: {tool_schema}
        Initial Arguments: {initial_args}
        
        Review the task and optimize the arguments to ensure all relevant parameters are used correctly.
        Pay special attention to filtering parameters that would help narrow down results to match the task.
        """
        try:
            function_model = os.getenv("FUNCTION_MODEL")
            response = call_llm(
                prompt,
                system_prompt=get_tool_args_system_prompt(),
                output_schema=OptimizedToolArgs,
                model_override=function_model,
            )
            # Handle case where LLM returns dict directly instead of OptimizedToolArgs
            if isinstance(response, dict):
                return response if response else initial_args
            return response.arguments
        except Exception as e:
            self.logger._log(f"Argument optimization failed: {e}, using original args")
            return initial_args

    def _field_allows_list(self, field_schema: dict) -> bool:
        """Check if a schema field accepts list values."""
        if not isinstance(field_schema, dict):
            return False
        if field_schema.get("type") == "array":
            return True
        for key in ("anyOf", "oneOf"):
            if key in field_schema and isinstance(field_schema[key], list):
                for option in field_schema[key]:
                    if isinstance(option, dict) and option.get("type") == "array":
                        return True
        return False

    def _expand_tool_args(self, tool, args: dict) -> list[dict]:
        """Expand list arguments into multiple tool calls when schema expects scalars."""
        if not tool or not args:
            return [args]
        tool_schema = tool.args_schema.schema() if hasattr(tool, "args_schema") and tool.args_schema else {}
        properties = tool_schema.get("properties", {}) if isinstance(tool_schema, dict) else {}
        list_fields = {}
        for key, value in args.items():
            if isinstance(value, list):
                field_schema = properties.get(key, {})
                if not self._field_allows_list(field_schema):
                    list_fields[key] = value
        if not list_fields:
            return [args]

        expanded = [{}]
        for key, value in args.items():
            if key in list_fields:
                new_expanded = []
                for item in value:
                    for base in expanded:
                        merged = dict(base)
                        merged[key] = item
                        new_expanded.append(merged)
                expanded = new_expanded
            else:
                for base in expanded:
                    base[key] = value
        return expanded

    def _normalize_tool_args(self, tool_name: str, args: dict) -> dict:
        """Normalize loose tool arguments (symbol vs. ticker, latest shortcuts, missing period)."""
        if not isinstance(args, dict):
            return args
        normalized = dict(args)

        # Normalize symbol/ticker parameter names (including plural forms)
        if "symbol" in normalized and "ticker" not in normalized:
            normalized["ticker"] = normalized.pop("symbol")
        if "tickers" in normalized and "ticker" not in normalized:
            # Handle plural form - if it's a single ticker, convert to singular
            tickers_value = normalized.pop("tickers")
            if isinstance(tickers_value, str):
                normalized["ticker"] = tickers_value
            elif isinstance(tickers_value, list) and len(tickers_value) == 1:
                normalized["ticker"] = tickers_value[0]
            else:
                # If multiple tickers, keep as list for expansion
                normalized["ticker"] = tickers_value

        statement_tools = {
            "get_income_statements", "get_balance_sheets", "get_cash_flow_statements",
            "yf_get_income_statements", "yf_get_balance_sheets", "yf_get_cash_flow_statements"
        }
        if tool_name in statement_tools:
            period = normalized.get("period")
            latest = normalized.pop("latest", None)
            if isinstance(period, str) and period.lower() == "latest":
                normalized["period"] = "quarterly"
            elif isinstance(latest, str):
                candidate = latest.lower()
                if candidate in {"annual", "quarterly", "ttm"}:
                    normalized["period"] = candidate
                elif candidate == "latest":
                    normalized.setdefault("period", "quarterly")
            elif latest:
                normalized.setdefault("period", "quarterly")
            elif not period:
                normalized.setdefault("period", "quarterly")

        if tool_name in {"get_price_snapshot", "yf_get_price_snapshot"}:
            normalized.setdefault("ticker", normalized.get("symbol"))

        return normalized

    # ---------- tool execution ----------
    def _execute_tool(self, tool, tool_name: str, inp_args):
        """Execute a tool with progress indication."""
        # Create a dynamic decorator with the tool name
        @show_progress(f"Executing {tool_name}...", "")
        def run_tool():
            return tool.run(inp_args)
        return run_tool()

    def _execute_tool_no_progress(self, tool, inp_args):
        """Execute a tool without UI progress (safe for parallel runs)."""
        return tool.run(inp_args)
    
    # ---------- confirm action ----------
    def confirm_action(self, tool: str, input_str: str) -> bool:
        # In production you'd ask the user; here we just log and auto-confirm
        # Risky tools are not implemented in this version.
        return True

    # ---------- main loop ----------
    def run(self, query: str):
        """
        Executes the main agent loop to process a user query.

        This method orchestrates the entire process of understanding a query,
        planning tasks, executing tools to gather information, and synthesizing
        a final answer.

        Args:
            query (str): The user's natural language query.

        Returns:
            str: A comprehensive answer to the user's query.
        """
        # Display the user's query
        self.logger.log_user_query(query)
        
        # Initialize agent state for this run.
        step_count = 0
        last_actions = []
        session_outputs = []

        # 1. Decompose the user query into a list of tasks.
        tasks = self.plan_tasks(query)

        # If no tasks were created, the query is likely out of scope.
        if not tasks:
            answer = self._generate_answer(query, session_outputs)
            self.logger.log_summary(answer, query=query)
            return answer

        # 2. Execute tasks until all are complete or max steps are reached.
        while any(not t.done for t in tasks):
            # Global safety break.
            if step_count >= self.max_steps:
                self.logger._log("Global max steps reached — aborting to avoid runaway loop.")
                break

            # Select the next incomplete task.
            task = next(t for t in tasks if not t.done)
            self.logger.log_task_start(task.description)

            # Loop for a single task, with its own step limit.
            per_task_steps = 0
            task_outputs = []
            while per_task_steps < self.max_steps_per_task:
                if step_count >= self.max_steps:
                    self.logger._log("Global max steps reached — stopping.")
                    return

                # Ask the LLM for the next action to take for the current task.
                ai_message = self.ask_for_actions(task.description, last_outputs="\n".join(task_outputs))
                
                # If no tool is called, the task is considered complete.
                if not ai_message.tool_calls:
                    task.done = True
                    self.logger.log_task_done(task.description)
                    break

                # Process each tool call returned by the LLM.
                for tool_call in ai_message.tool_calls:
                    if step_count >= self.max_steps:
                        break

                    tool_name = tool_call["name"]
                    initial_args = tool_call["args"]
                    
                    # Refine tool arguments for better performance, unless disabled.
                    optimized_args = (
                        initial_args
                        if self.disable_tool_arg_optimization
                        else self.optimize_tool_args(tool_name, initial_args, task.description)
                    )
                    normalized_args = self._normalize_tool_args(tool_name, optimized_args)

                    # Execute the tool.
                    tool_to_run = next((t for t in self.tools if t.name == tool_name), None)
                    expanded_args_list = self._expand_tool_args(tool_to_run, normalized_args)
                    to_execute = []
                    for expanded_args in expanded_args_list:
                        # Create a signature of the action to be taken.
                        action_sig = f"{tool_name}:{expanded_args}"

                        # Detect and prevent repetitive action loops.
                        last_actions.append(action_sig)
                        if len(last_actions) > 4:
                            last_actions = last_actions[-4:]
                        if len(set(last_actions)) == 1 and len(last_actions) == 4:
                            self.logger._log("Detected repeating action — aborting to avoid loop.")
                            return

                        if tool_to_run and self.confirm_action(tool_name, str(expanded_args)):
                            to_execute.append((tool_to_run, tool_name, expanded_args))
                        else:
                            self.logger._log(f"Invalid tool: {tool_name}")

                    if to_execute:
                        max_workers = min(self.max_parallel_tool_calls, len(to_execute))
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = {
                                executor.submit(self._execute_tool_no_progress, tool_obj, args): (name, args)
                                for tool_obj, name, args in to_execute
                            }
                            for future in as_completed(futures):
                                name, args = futures[future]
                                try:
                                    result = future.result()
                                    self.logger.log_tool_run(name, f"{result}")
                                    output = f"Output of {name} with args {args}: {result}"
                                    session_outputs.append(output)
                                    task_outputs.append(output)
                                except Exception as e:
                                    self.logger._log(f"Tool execution failed: {e}")
                                    error_output = f"Error from {name} with args {args}: {e}"
                                    session_outputs.append(error_output)
                                    task_outputs.append(error_output)

                    step_count += len(to_execute)
                    per_task_steps += len(to_execute)

                # After a batch of tool calls, check if the task is complete.
                if self.ask_if_done(task.description, "\n".join(task_outputs)):
                    task.done = True
                    self.logger.log_task_done(task.description)
                    break

        # 3. Synthesize the final answer from all collected tool outputs.
        answer = self._generate_answer(query, session_outputs)
        self.logger.log_summary(answer, query=query)
        return answer
    
    # ---------- answer generation ----------
    @show_progress("Generating answer...", "Answer ready")
    def _generate_answer(self, query: str, session_outputs: list) -> str:
        """Generate the final answer based on collected data."""
        all_results = "\n\n".join(session_outputs) if session_outputs else "No data was collected."
        answer_prompt = f"""
        Original user query: "{query}"
        
        Data and results collected from tools:
        {all_results}
        
        Based on the data above, provide a comprehensive answer to the user's query.
        Include specific numbers, calculations, and insights.
        """
        answer_obj = call_llm(answer_prompt, system_prompt=get_answer_system_prompt(), output_schema=Answer)
        return answer_obj.answer
