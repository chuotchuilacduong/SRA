"""LLM wrapper supporting OpenAI-compatible API and Timely API.

Priority:
  1. Timely  — when TIMELY_API_KEY is present in env/.env (and no explicit api_base given)
  2. OpenAI  — when OPENAI_API_KEY / OPENAI_API_BASE is set, or api_base is passed

Config via CLI args (--model, --api-base) and env vars.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the SR-Agents project root (SR-Agents/.env) on first import.
load_dotenv(Path(__file__).parents[2] / ".env", override=False)


# ---------------------------------------------------------------------------
# Timely duck-typed adapter (mirrors OpenAI client's chat.completions.create)
# ---------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _TimelyCompletions:
    def __init__(self, timely_client) -> None:
        self._c = timely_client

    def create(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop=None,
        extra_body=None,
        **_kwargs,
    ) -> _FakeResponse:
        content = self._c.call(messages, model=model)
        return _FakeResponse(content or "")


class _TimelyChat:
    def __init__(self, timely_client) -> None:
        self.completions = _TimelyCompletions(timely_client)


class TimelyLLMClient:
    """Duck-typed drop-in for ``openai.OpenAI`` — routes calls to Timely API."""

    def __init__(self, api_key: str | None = None) -> None:
        from sragents.timely_client import TimelyClient
        self.chat = _TimelyChat(TimelyClient(api_key=api_key))


def create_llm_client(
    api_base: str | None = None, api_key: str | None = None
) -> "OpenAI | TimelyLLMClient":
    """Return a Timely client if TIMELY_API_KEY is set, else OpenAI client."""
    timely_key = os.environ.get("TIMELY_API_KEY")
    if timely_key and api_base is None:
        return TimelyLLMClient(api_key=timely_key)

    base_url = api_base or os.environ.get("OPENAI_API_BASE")
    key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    return OpenAI(base_url=base_url, api_key=key)


_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from an LLM response.

    Handles both well-formed `<think>...</think>answer` and truncation
    cases where generation is cut mid-thinking (unclosed tag).
    """
    if "<think>" not in text:
        return text
    text = _THINK_CLOSED_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.lstrip()


def get_extra_body(model: str, thinking: bool = False) -> dict | None:
    """Return per-model extra_body for thinking/reasoning control.

    By default (thinking=False) suppresses thinking on hybrid-thinking
    models so they're comparable to non-reasoning baselines. GPT-5 is a
    pure reasoning model; we always run it at minimal effort.

      - Qwen3: chat_template_kwargs.enable_thinking
      - GLM-5 / Kimi: enable_thinking
      - GPT-5: reasoning_effort="minimal" (always)
      - Others (Llama, Mistral, MiniMax, ...): no flag
    """
    basename = model.lower().rsplit("/", 1)[-1]

    if "qwen3" in basename:
        return {"chat_template_kwargs": {"enable_thinking": thinking}}
    if "gpt-5" in basename:
        return {"reasoning_effort": "minimal"}
    if "glm-5" in basename or "kimi" in basename:
        return {"enable_thinking": thinking}
    return None


def chat(
    client: OpenAI,
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stop: list[str] | None = None,
    extra_body: dict | None = None,
) -> str:
    """Send chat completion request, return content string."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if stop:
        kwargs["stop"] = stop
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def chat_messages(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stop: list[str] | None = None,
    extra_body: dict | None = None,
) -> str:
    """Send chat completion with an explicit messages list (for multi-turn)."""
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if stop:
        kwargs["stop"] = stop
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content
