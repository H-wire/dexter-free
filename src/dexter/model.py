import os
import re
import json
import time
import urllib.request
import urllib.error
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
_llm_instances: dict[tuple, ChatOpenAI] = {}


def _get_num_ctx_for_model(model_name: str) -> Optional[int]:
    """Resolve context window based on the selected model."""
    function_model = os.getenv("FUNCTION_MODEL")
    if function_model and model_name == function_model:
        num_ctx_env = os.getenv("FUNCTION_NUM_CTX")
    else:
        num_ctx_env = os.getenv("OPENAI_NUM_CTX")
    if not num_ctx_env:
        return None
    try:
        return int(num_ctx_env)
    except ValueError:
        _logger._log("Invalid OPENAI_NUM_CTX/FUNCTION_NUM_CTX; ignoring.")
        return None


def _get_base_extra_body(base_url: Optional[str], model_name: str) -> Optional[dict]:
    """Build extra_body for Ollama/OpenAI-compatible servers."""
    if not _is_ollama_endpoint(base_url):
        return None
    num_ctx = _get_num_ctx_for_model(model_name)
    if num_ctx is None:
        return None
    return {"options": {"num_ctx": num_ctx}}


def _ollama_api_base(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return None
    if base_url.endswith("/v1"):
        return base_url[:-3]
    return base_url


def _call_ollama_chat_json(
    prompt: str,
    system_prompt: str,
    model_name: str,
    output_schema: Type[BaseModel],
) -> Optional[BaseModel]:
    base_url = os.getenv("OPENAI_BASE_URL")
    api_base = _ollama_api_base(base_url)
    if not api_base:
        return None
    num_ctx = _get_num_ctx_for_model(model_name)
    options: dict[str, Any] = {"temperature": 0}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    payload = {
        "model": model_name,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "options": options,
        "stream": False,
    }
    url = f"{api_base}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    timeout_env = os.getenv("OPENAI_TIMEOUT")
    try:
        timeout = float(timeout_env) if timeout_env else None
    except ValueError:
        timeout = None

    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except Exception as e:
        _logger._log(f"Ollama JSON bypass failed: {e}")
        return None

    try:
        payload = json.loads(body)
    except Exception:
        return None

    content = payload.get("message", {}).get("content")
    if not isinstance(content, str):
        return None
    data = _extract_json(content)
    if not isinstance(data, dict):
        return None
    try:
        return output_schema.model_validate(data)
    except Exception:
        return None


def get_llm(model_override: Optional[str] = None) -> ChatOpenAI:
    """Create or return a cached ChatOpenAI instance based on environment.

    Respects the following environment variables:
    - OPENAI_BASE_URL: Custom endpoint (e.g., http://localhost:1234/v1)
    - OPENAI_MODEL: Model name/id (default: gpt-4.1)
    - OPENAI_API_KEY: Optional; if missing and base URL is set, default to 'sk-local'
    - OPENAI_TIMEOUT: Per-request timeout in seconds (optional)
    - OPENAI_MAX_RETRIES: LLM internal retry count (optional)
    """
    base_url = os.getenv("OPENAI_BASE_URL")
    model = model_override or os.getenv("OPENAI_MODEL", "gpt-4.1")
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

    extra_body = _get_base_extra_body(base_url, model)
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
    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    extra_body_key = json.dumps(extra_body, sort_keys=True) if extra_body else None
    cache_key = (model, base_url, api_key, timeout, max_retries, extra_body_key)
    if cache_key in _llm_instances:
        return _llm_instances[cache_key]

    instance = ChatOpenAI(**kwargs)
    _llm_instances[cache_key] = instance
    if base_url:
        _logger._log(f"LLM configured for custom base URL: {base_url} (model={model})")
    else:
        _logger._log(f"LLM configured for OpenAI cloud (default) (model={model})")
    return instance


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


def _truncate_prompt(text: str, max_chars: Optional[int]) -> tuple[str, bool]:
    if not max_chars or max_chars <= 0:
        return text, False
    if len(text) <= max_chars:
        return text, False
    return f"{text[:max_chars]}...[truncated]", True


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
    model_override: Optional[str] = None,
) -> AIMessage:
    """Call the LLM with optional structured output and tools.

    Falls back to plain text if the backend does not support
    function-calling or tool-binding (common on local servers).
    """
    max_prompt_chars_env = os.getenv("DEXTER_MAX_PROMPT_CHARS")
    try:
        max_prompt_chars = int(max_prompt_chars_env) if max_prompt_chars_env else None
    except ValueError:
        _logger._log("Invalid DEXTER_MAX_PROMPT_CHARS; ignoring.")
        max_prompt_chars = None

    final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
    final_system_prompt, system_truncated = _truncate_prompt(final_system_prompt, max_prompt_chars)
    prompt, user_truncated = _truncate_prompt(prompt, max_prompt_chars)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", final_system_prompt),
        ("user", "{prompt}"),
    ])

    llm = get_llm(model_override=model_override)
    runnable = llm
    prefer_tools = False
    base_url = os.getenv("OPENAI_BASE_URL")
    json_mode = False
    base_extra_body = _get_base_extra_body(base_url, llm.model_name)
    function_model = os.getenv("FUNCTION_MODEL")
    use_ollama_bypass = (
        output_schema is not None
        and function_model
        and model_override == function_model
        and _is_ollama_endpoint(base_url)
    )

    try:
        if output_schema is not None and _is_ollama_endpoint(base_url) and not tools:
            json_mode = True
            extra_body = dict(base_extra_body or {})
            extra_body["format"] = "json"
            runnable = llm.bind(extra_body=extra_body)
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
    meta_bits.append(f"model={llm.model_name}")
    if json_mode:
        meta_bits.append("json_mode=ollama")
    if use_ollama_bypass:
        meta_bits.append("ollama_bypass=api_chat")
    if system_truncated or user_truncated:
        meta_bits.append("prompt_truncated=true")
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
            if use_ollama_bypass:
                result = _call_ollama_chat_json(
                    prompt=prompt,
                    system_prompt=final_system_prompt,
                    model_name=llm.model_name,
                    output_schema=output_schema,
                )
                if result is not None:
                    _logger.log_llm_response(
                        _truncate_text(_format_llm_response(result)),
                        meta=meta,
                    )
                    return result  # type: ignore[return-value]
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
