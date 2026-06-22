import json
import os
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

from .messages import (
    _rough_image_tokens,
    _rough_message_tokens,
    _rough_text_tokens,
    normalize_messages,
    to_antrophic,
)


def _content_debug_stats(content: Any) -> dict[str, int]:
    stats = {
        "text_chars": 0,
        "rough_text_tokens": 0,
        "image_count": 0,
        "image_base64_chars": 0,
        "rough_image_tokens": 0,
    }
    if isinstance(content, str):
        stats["text_chars"] += len(content)
        stats["rough_text_tokens"] += _rough_text_tokens(content)
        return stats

    if not isinstance(content, list):
        return stats

    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text = str(part.get("text", "") or "")
            stats["text_chars"] += len(text)
            stats["rough_text_tokens"] += _rough_text_tokens(text)
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url") or ""
            stats["image_count"] += 1
            if isinstance(url, str) and url.startswith("data:image") and "," in url:
                stats["image_base64_chars"] += len(url.split(",", 1)[1])
            stats["rough_image_tokens"] += _rough_image_tokens(url)
    return stats


def _sum_stats(rows: list[dict[str, int]]) -> dict[str, int]:
    keys = [
        "text_chars",
        "rough_text_tokens",
        "image_count",
        "image_base64_chars",
        "rough_image_tokens",
    ]
    return {key: sum(row.get(key, 0) for row in rows) for key in keys}


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
        self._debug_tokens = os.getenv("MAIA_DEBUG_TOKENS", "0") == "1"

    def _print_token_debug(
        self,
        raw_messages: list[dict[str, Any]],
        norm_messages: list[dict[str, Any]],
        *,
        max_output_tokens: int,
        attempt: int,
    ) -> None:
        if not self._debug_tokens:
            return

        raw_system_stats = _sum_stats(
            [
                _content_debug_stats(message.get("content"))
                for message in raw_messages
                if message.get("role") == "system"
            ]
        )
        raw_non_system_stats = _sum_stats(
            [
                _content_debug_stats(message.get("content"))
                for message in raw_messages
                if message.get("role") != "system"
            ]
        )
        norm_stats_by_message = [
            {
                **_content_debug_stats(message.get("content")),
                "rough_total_tokens": _rough_message_tokens(message),
                "role": str(message.get("role")),
            }
            for message in norm_messages
        ]
        norm_total = _sum_stats(norm_stats_by_message)
        rough_norm_message_tokens = sum(row["rough_total_tokens"] for row in norm_stats_by_message)

        print(
            "[MAIA_TOKEN_DEBUG] local request "
            f"model={self.model} attempt={attempt} max_output_tokens={max_output_tokens} "
            f"raw_messages={len(raw_messages)} normalized_messages={len(norm_messages)}"
        )
        print(
            "[MAIA_TOKEN_DEBUG] raw_system "
            f"rough_text_tokens={raw_system_stats['rough_text_tokens']} "
            f"text_chars={raw_system_stats['text_chars']} "
            f"images={raw_system_stats['image_count']} "
            f"rough_image_tokens={raw_system_stats['rough_image_tokens']}"
        )
        print(
            "[MAIA_TOKEN_DEBUG] raw_non_system "
            f"rough_text_tokens={raw_non_system_stats['rough_text_tokens']} "
            f"text_chars={raw_non_system_stats['text_chars']} "
            f"images={raw_non_system_stats['image_count']} "
            f"rough_image_tokens={raw_non_system_stats['rough_image_tokens']}"
        )
        print(
            "[MAIA_TOKEN_DEBUG] normalized_payload "
            f"rough_message_tokens={rough_norm_message_tokens} "
            f"rough_text_tokens={norm_total['rough_text_tokens']} "
            f"images={norm_total['image_count']} "
            f"image_base64_chars={norm_total['image_base64_chars']} "
            f"rough_image_tokens={norm_total['rough_image_tokens']}"
        )
        for idx, row in enumerate(norm_stats_by_message):
            print(
                "[MAIA_TOKEN_DEBUG] normalized_message "
                f"index={idx} role={row['role']} "
                f"rough_total_tokens={row['rough_total_tokens']} "
                f"rough_text_tokens={row['rough_text_tokens']} "
                f"text_chars={row['text_chars']} images={row['image_count']} "
                f"rough_image_tokens={row['rough_image_tokens']}"
            )

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
        min_retry_tokens = 16
        if absolute_room < min_retry_tokens:
            return None

        # Be deliberately conservative. vLLM's reported prompt-token count can
        # shift slightly between retries for multimodal/chat-template requests,
        # and reducing by only a few tokens can burn all retries near the limit.
        if absolute_room >= 192:
            safe_by_room = absolute_room - 128
        else:
            safe_by_room = max(min_retry_tokens, absolute_room - 8)
        aggressive_step = int(requested_tokens * 0.75)
        retry_tokens = min(requested_tokens - 1, safe_by_room, aggressive_step)
        if retry_tokens < min_retry_tokens or retry_tokens >= requested_tokens:
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
            self._print_token_debug(
                messages,
                norm_messages,
                max_output_tokens=int(params["max_tokens"]),
                attempt=_attempt + 1,
            )
            request = Request(
                url,
                data=json.dumps(params).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(request, timeout=300) as response:
                    resp = json.loads(response.read().decode("utf-8"))
                if self._debug_tokens and isinstance(resp.get("usage"), dict):
                    usage = resp["usage"]
                    print(
                        "[MAIA_TOKEN_DEBUG] response_usage "
                        f"prompt_tokens={usage.get('prompt_tokens')} "
                        f"completion_tokens={usage.get('completion_tokens')} "
                        f"total_tokens={usage.get('total_tokens')} "
                        f"details={usage}"
                    )
                return resp["choices"][0]["message"]["content"]
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {body}"
                retry_tokens = self._retry_tokens_after_context_error(
                    body, int(params["max_tokens"])
                )
                if self._debug_tokens:
                    print(f"[MAIA_TOKEN_DEBUG] http_error status={exc.code} body={body}")
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
