import os
import re
import json
import time
from typing import Type, List, Optional, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from openai import APIConnectionError
from pydantic import BaseModel

from dexter.prompts import DEFAULT_SYSTEM_PROMPT
from dexter.schemas import Answer
from dexter.utils.logger import Logger

_logger = Logger()
_llm_instance: Optional[ChatOpenAI] = None


def get_llm() -> ChatOpenAI:
    """Create or return a cached ChatOpenAI instance based on environment.

    Respects the following environment variables:
    - OPENAI_BASE_URL: Custom endpoint (e.g., http://localhost:1234/v1)
    - OPENAI_MODEL: Model name/id (default: gpt-4.1)
    - OPENAI_API_KEY: Optional; if missing and base URL is set, default to 'sk-local'
    - OPENAI_TIMEOUT: Per-request timeout in seconds (optional)
    - OPENAI_MAX_RETRIES: LLM internal retry count (optional)
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")
    api_key = os.getenv("OPENAI_API_KEY")
    timeout_env = os.getenv("OPENAI_TIMEOUT")
    max_retries_env = os.getenv("OPENAI_MAX_RETRIES")

    # For local servers that don't require keys, supply a benign default
    if base_url and not api_key:
        api_key = "sk-local"

    timeout = None
    try:
        timeout = float(timeout_env) if timeout_env else None
    except ValueError:
        _logger._log("Invalid OPENAI_TIMEOUT; ignoring.")

    max_retries = None
    try:
        max_retries = int(max_retries_env) if max_retries_env else None
    except ValueError:
        _logger._log("Invalid OPENAI_MAX_RETRIES; ignoring.")

    kwargs = {
        "model": model,
        "temperature": 0,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries

    _llm_instance = ChatOpenAI(**kwargs)
    if base_url:
        _logger._log(f"LLM configured for custom base URL: {base_url}")
    else:
        _logger._log("LLM configured for OpenAI cloud (default)")
    return _llm_instance


def _extract_json(text: str) -> Optional[Any]:
    """Try to parse JSON from a text blob, handling code fences."""
    if not text:
        return None
    # Remove common code fences
    fenced = re.search(r"```(?:json)?\n(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        return None


def _truncate_text(text: str, limit: int = 2000) -> str:
    """Trim long text for logs while preserving a hint of truncation."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _format_llm_response(result: Any) -> str:
    """Best-effort string formatting for LLM responses."""
    if isinstance(result, BaseModel):
        try:
            return result.model_dump_json()
        except Exception:
            return str(result)
    if isinstance(result, AIMessage):
        return str(getattr(result, "content", ""))
    return str(result)


def _is_structured_result(result: Any) -> bool:
    """Return True when a parsed schema object (not a raw AIMessage)."""
    return isinstance(result, BaseModel) and not isinstance(result, AIMessage)


def _is_ollama_endpoint(base_url: Optional[str]) -> bool:
    """Heuristic to detect Ollama OpenAI-compatible endpoints."""
    if not base_url:
        return False
    lowered = base_url.lower()
    return "ollama" in lowered or "11434" in lowered


def _is_empty_answer(answer: str) -> bool:
    """Detect empty or placeholder Answer content."""
    normalized = (answer or "").strip().lower()
    return normalized in {"", "none", "null"}


def _fallback_plain_answer(
    prompt_template: ChatPromptTemplate,
    llm: ChatOpenAI,
    prompt: str,
    meta: str,
) -> Answer:
    """Retry without structured output and coerce into Answer."""
    fallback_chain = prompt_template | llm
    fallback = fallback_chain.invoke({"prompt": prompt})
    content = getattr(fallback, "content", str(fallback))
    coerced = Answer(answer=str(content))
    _logger.log_llm_response(
        _truncate_text(_format_llm_response(coerced)),
        meta=f"{meta} fallback=plain",
    )
    return coerced


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    tools: Optional[List[BaseTool]] = None,
) -> AIMessage:
    """Call the LLM with optional structured output and tools.

    Falls back to plain text if the backend does not support
    function-calling or tool-binding (common on local servers).
    """
    final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", final_system_prompt),
        ("user", "{prompt}"),
    ])

    llm = get_llm()
    runnable = llm
    prefer_tools = False
    base_url = os.getenv("OPENAI_BASE_URL")
    json_mode = False

    try:
        if output_schema is not None and _is_ollama_endpoint(base_url) and not tools:
            json_mode = True
            runnable = llm.bind(extra_body={"format": "json"})
        elif output_schema is not None:
            runnable = llm.with_structured_output(output_schema, method="function_calling")
        elif tools:
            prefer_tools = True
            runnable = llm.bind_tools(tools)
    except Exception as e:
        _logger._log(f"LLM does not support structured outputs/tools: {e}. Falling back to plain model.")
        runnable = llm

    chain = prompt_template | runnable
    tool_names = [t.name for t in tools] if tools else []
    meta_bits = []
    if output_schema is not None:
        meta_bits.append(f"schema={output_schema.__name__}")
    if tool_names:
        meta_bits.append(f"tools={tool_names}")
    if json_mode:
        meta_bits.append("json_mode=ollama")
    meta = " ".join(meta_bits)
    _logger.log_llm_request(
        _truncate_text(final_system_prompt),
        _truncate_text(prompt),
        meta=meta,
    )

    # Retry logic for transient connection errors; configurable via env
    max_attempts_env = os.getenv("OPENAI_MAX_RETRIES")
    try:
        max_attempts = int(max_attempts_env) if max_attempts_env else 3
    except ValueError:
        max_attempts = 3

    for attempt in range(max_attempts):
        try:
            result = chain.invoke({"prompt": prompt})
            # If structured output is requested but not returned, try to coerce
            if output_schema is not None and not _is_structured_result(result):
                if json_mode:
                    content = getattr(result, "content", None)
                    data = _extract_json(content) if isinstance(content, str) else None
                    if data is not None:
                        try:
                            coerced = output_schema.model_validate(data)
                            _logger.log_llm_response(
                                _truncate_text(_format_llm_response(coerced)),
                                meta=meta,
                            )
                            if output_schema is Answer and _is_empty_answer(coerced.answer):
                                return _fallback_plain_answer(prompt_template, llm, prompt, meta)
                            return coerced  # type: ignore[return-value]
                        except Exception:
                            pass
                # Special-case Answer schema to avoid crashes
                if output_schema is Answer:
                    content = getattr(result, "content", str(result))
                    try:
                        coerced = Answer(answer=str(content))
                        _logger.log_llm_response(
                            _truncate_text(_format_llm_response(coerced)),
                            meta=meta,
                        )
                        if _is_empty_answer(coerced.answer):
                            return _fallback_plain_answer(prompt_template, llm, prompt, meta)
                        return coerced  # type: ignore[return-value]
                    except Exception:
                        return result
                # Try JSON coercion for other schemas
                content = getattr(result, "content", None)
                data = _extract_json(content) if isinstance(content, str) else None
                if data is not None:
                    try:
                        coerced = output_schema.model_validate(data)
                        _logger.log_llm_response(
                            _truncate_text(_format_llm_response(coerced)),
                            meta=meta,
                        )
                        return coerced  # type: ignore[return-value]
                    except Exception:
                        pass
            _logger.log_llm_response(
                _truncate_text(_format_llm_response(result)),
                meta=meta,
            )
            if output_schema is Answer and isinstance(result, Answer):
                if _is_empty_answer(result.answer):
                    return _fallback_plain_answer(prompt_template, llm, prompt, meta)
            if output_schema is Answer and not isinstance(result, Answer):
                content = getattr(result, "content", None)
                if isinstance(content, str):
                    data = _extract_json(content)
                    if isinstance(data, dict):
                        if _is_empty_answer(str(data.get("answer", ""))):
                            return _fallback_plain_answer(prompt_template, llm, prompt, meta)
                    if _is_empty_answer(content):
                        return _fallback_plain_answer(prompt_template, llm, prompt, meta)
            return result
        except APIConnectionError as e:
            if attempt >= max_attempts - 1:
                raise
            time.sleep(0.5 * (2 ** attempt))
        except Exception as e:
            # If tool-calling path failed, try a final plain LLM pass
            if prefer_tools or output_schema is not None:
                _logger._log(f"Invocation failed; retrying without tools/structured output: {e}")
                chain = prompt_template | llm
                prefer_tools = False
                output_schema = None
                continue
            raise
