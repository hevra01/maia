import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import anthropic
from anthropic.types import Message

try:
    import openai
except ImportError:
    openai = None

from .messages import normalize_messages, to_antrophic


class OpenAIAdapter:
    """
    Uses OpenAI's ChatCompletion (compatible with many proxies).
    Accepts OpenAI-style `messages`.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        organization: str | None = None,
        base_url: str | None = None,
    ):
        if openai is None:
            raise ImportError(
                "The openai package is required only when using AGENT=gpt-*. "
                "Use AGENT=local-<model> for vLLM without installing openai."
            )
        openai.api_key = api_key
        if organization:
            openai.organization = organization
        if base_url:
            openai.api_base = base_url
        self.model = model
        self._allow_system = True  # OpenAI supports system

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_output_tokens: int,
        **kwargs,
    ) -> str:
        # Normalize but keep system messages for OpenAI
        norm_messages = normalize_messages(messages, allow_system=self._allow_system)
        if not norm_messages:
            raise ValueError("After normalization, no valid user/assistant messages remain.")
        params: dict[str, Any] = {
            "model": self.model,
            "messages": norm_messages,
            "max_tokens": max_output_tokens,
        }
        params.update(kwargs)
        resp: dict[str, Any] = openai.ChatCompletion.create(**params)
        return resp["choices"][0]["message"]["content"]


class LocalAdapter:
    """
    For OpenAI-compatible local servers (vLLM, Ollama bridges, etc.).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3",
        api_key: str = "dummy",
        allow_system: bool = False,  # Gemma / many local backends don't support system
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._allow_system = allow_system

    @staticmethod
    def _retry_tokens_after_context_error(body: str, requested_tokens: int) -> int | None:
        """
        Parse vLLM/OpenAI-compatible context-overflow errors and return a
        smaller output-token budget for one retry.
        """
        try:
            payload = json.loads(body)
            message = payload.get("error", {}).get("message", "")
        except Exception:
            message = body

        context_match = re.search(r"maximum context length is (\d+) tokens", message)
        input_match = re.search(r"prompt contains at least (\d+) input tokens", message)
        if context_match is None or input_match is None:
            return None

        context_tokens = int(context_match.group(1))
        input_tokens = int(input_match.group(1))
        absolute_room = context_tokens - input_tokens
        if absolute_room < 64:
            return None

        # Be deliberately conservative. vLLM's reported prompt-token count can
        # shift slightly between retries for multimodal/chat-template requests,
        # and reducing by only a few tokens can burn all retries near the limit.
        if absolute_room >= 192:
            safe_by_room = absolute_room - 128
        else:
            safe_by_room = absolute_room // 2
        aggressive_step = int(requested_tokens * 0.75)
        retry_tokens = min(requested_tokens - 1, safe_by_room, aggressive_step)
        if retry_tokens < 64 or retry_tokens >= requested_tokens:
            return None
        return retry_tokens

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_output_tokens: int,
        **kwargs,
    ) -> str:
        # Strip/handle system according to local model capability
        norm_messages = normalize_messages(
            messages, merge_system_into_user=True, keep_system=self._allow_system
        )
        if not norm_messages:
            raise ValueError(
                "After normalization, no valid user/assistant messages remain for local model."
            )

        params: dict[str, Any] = {
            "model": self.model,
            "messages": norm_messages,
            "max_tokens": max_output_tokens,
        }
        params.update(kwargs)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: str | None = None
        for _attempt in range(8):
            request = Request(
                url,
                data=json.dumps(params).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(request, timeout=300) as response:
                    resp = json.loads(response.read().decode("utf-8"))
                return resp["choices"][0]["message"]["content"]
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {body}"
                retry_tokens = self._retry_tokens_after_context_error(
                    body, int(params["max_tokens"])
                )
                if retry_tokens is not None:
                    print(
                        "Local model context window is tight; retrying with "
                        f"max_tokens={retry_tokens} instead of {params['max_tokens']}."
                    )
                    params["max_tokens"] = retry_tokens
                    continue
                raise RuntimeError(f"Local model request failed: HTTP {exc.code}: {body}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                raise RuntimeError(
                    "Local model request failed before receiving an HTTP response. "
                    f"URL={url}; error={exc!r}"
                ) from exc
        raise RuntimeError(
            "Local model request failed after exhausting context-window retries. "
            f"URL={url}; final max_tokens={params['max_tokens']}; last_error={last_error}"
        )


# Anhtropic (Claude Sonnet 4)
class AnthropicAdapter:
    """
    Adapts OpenAI-style messages to Anthropic's.
    """

    def __init__(self, api_key: str, model: str = "claude-4-sonnet-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _split(self, messages: list[dict[str, Any]]):
        system = None
        history: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            blocks = to_antrophic(m)
            if role == "system":
                system = blocks or [{"type": "text", "text": ""}]
            else:
                history.append(
                    {
                        "role": "user" if role == "user" else "assistant",
                        "content": blocks or [{"type": "text", "text": ""}],
                    }
                )
        return system, history

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_output_tokens: int,
        **kwargs,
    ) -> str:
        system, messages = self._split(messages)
        params = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_output_tokens,
        }
        params.update(kwargs)

        resp: Message = self.client.messages.create(**params)
        return "".join(c.text for c in resp.content if getattr(c, "type", None) == "text")
