import json
import os
import time

from dexter.utils.ui import UI


class Logger:
    """Logger that uses the new interactive UI system."""
    
    def __init__(self):
        self.ui = UI()
        self.log = []
        logging_env = os.getenv("LOGGING", "false").strip().lower()
        self.logging_enabled = logging_env in {"1", "true", "yes", "on"}
        self.log_file = os.getenv("DEXTER_LOG_FILE", "dexter.log")

    def _log(self, msg: str):
        """Print immediately and keep in log."""
        print(msg, flush=True)
        self.log.append(msg)

    def _log_to_file(self, msg: str):
        """Write a log message to file if file logging is enabled."""
        if not self.logging_enabled:
            return
        try:
            with open(self.log_file, "a", encoding="utf-8") as log_handle:
                log_handle.write(msg + "\n")
        except Exception:
            pass

    def log_header(self, msg: str):
        self.ui.print_header(msg)
    
    def log_user_query(self, query: str):
        self.ui.print_user_query(query)

    def log_task_list(self, tasks):
        self.ui.print_task_list(tasks)

    def log_task_start(self, task_desc: str):
        self.ui.print_task_start(task_desc)

    def log_task_done(self, task_desc: str):
        self.ui.print_task_done(task_desc)

    def log_tool_run(self, tool: str, result: str = ""):
        self.ui.print_tool_run(tool, str(result)[:100])

    def log_risky(self, tool: str, input_str: str):
        self.ui.print_warning(f"Risky action {tool}({input_str}) — auto-confirmed")

    def log_summary(self, summary: str, query: str = ""):
        self.ui.print_answer(summary, query=query)
    
    def progress(self, message: str, success_message: str = ""):
        """Return a progress context manager for showing loading states."""
        return self.ui.progress(message, success_message)

    def log_llm_request(self, system_prompt: str, prompt: str, meta: str = ""):
        """Log outbound LLM request details."""
        record = {
            "event": "llm_request",
            "timestamp": time.time(),
            "system_prompt": system_prompt,
            "prompt": prompt,
            "meta": meta,
        }
        self._log_to_file(json.dumps(record, ensure_ascii=True))

    def log_llm_response(self, response: str, meta: str = ""):
        """Log inbound LLM response details."""
        record = {
            "event": "llm_response",
            "timestamp": time.time(),
            "response": response,
            "meta": meta,
        }
        self._log_to_file(json.dumps(record, ensure_ascii=True))
