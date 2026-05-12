"""Timely API client for SR-Agents LLM calls.

Ported from SR-Agents/request.py — trimmed to LLM-call functionality only.
Embedding support removed (use separate SentenceTransformer directly).
Thread-safe token cache via a lock (runner.py uses 32 threads sharing one client).
"""
import logging
import os
import threading
import time
import uuid

import requests

logger = logging.getLogger(__name__)


class TimelyClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.getenv("TIMELY_API_KEY")
        self.base_url = base_url or os.getenv(
            "TIMELY_BASE_URL", "https://hello.timelygpt.co.kr/api/v2/chat"
        )
        if not self.api_key:
            raise ValueError(
                "TIMELY_API_KEY not set — add it to .env or pass api_key="
            )
        self.model_name = model_name
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0
        self._lock = threading.Lock()

    # --- Auth ----------------------------------------------------------

    def _ensure_auth(self) -> None:
        with self._lock:
            if self._access_token and time.time() < self._token_expires_at:
                return
            url = f"{self.base_url}/sdk-auth/authenticate"
            headers = {
                "Content-Type": "application/json",
                "X-Timely-API": self.api_key,
            }
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            token = data.get("data", {}).get("access_token")
            if not data.get("success") or not token:
                raise RuntimeError(f"Timely auth failed: {data}")
            self._access_token = token
            self._token_expires_at = time.time() + 55 * 60
            logger.debug("Timely access token refreshed")

    # --- Core call -----------------------------------------------------

    def call(
        self,
        messages: list[dict],
        model: str | None = None,
        _retry: bool = True,
    ) -> str:
        """Send a chat request with an OpenAI-style messages list.

        Each message is ``{"role": "user"|"system"|"assistant", "content": "..."}``.
        A fresh session_id is generated per call so parallel workers don't
        share conversation state.
        """
        self._ensure_auth()
        url = f"{self.base_url}/llm-completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._access_token}",
        }
        payload = {
            "session_id": f"sragents_{uuid.uuid4()}",
            "messages": messages,
            "chat_model_node": {"model": model or self.model_name},
            "chat_type": "DYNAMIC_CHAT",
            "stream": False,
            "locale": "vi",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=120)

        if resp.status_code == 401:
            with self._lock:
                self._access_token = None
            if _retry:
                return self.call(messages, model=model, _retry=False)
            raise RuntimeError("Timely: token refresh failed on retry")

        resp.raise_for_status()
        return self._parse_response(resp.json())

    def _parse_response(self, data: dict) -> str:
        msg_type = data.get("type")
        if msg_type == "final_response":
            return data.get("message", "")
        if msg_type == "error":
            raise RuntimeError(f"Timely API error: {data.get('error')}")
        logger.warning("Unexpected Timely response type %r: %s", msg_type, data)
        return str(data)
