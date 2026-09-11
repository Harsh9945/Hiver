# src/llm.py
import os
import sys
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import hashlib
import json
import threading
import time

class QuotaExhaustedError(RuntimeError):
    """Raised when the LLM provider API quota is completely exhausted."""
    pass

class LLMClient:
    """
    Resilient multi-key LLM client supporting:
    1. OpenAI API (defaulting to gpt-4o-mini).
    2. Google Gemini via OpenAI-compatible endpoint.
    3. Multi-key pool rotation with circuit breaker on quota exhaustion.
    4. In-memory SHA-256 prompt caching.
    5. Local open-weight models via Ollama.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.lock = threading.Lock()
        
        raw_keys = os.getenv("OPENAI_API_KEYS", "")
        if raw_keys:
            self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        else:
            single = api_key or os.getenv("OPENAI_API_KEY", "")
            self.api_keys = [single] if single else []

        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", None)
        self.model = model or os.getenv("OPENAI_MODEL", "gemini-3.5-flash-lite")
        self.key_idx = 0
        
        self.clients = []
        self.exhausted_keys = set()
        self.circuit_open = False
        self.is_available = False
        self._cache = {}
        
        try:
            from openai import OpenAI
            if self.api_keys or self.base_url:
                if not self.api_keys and self.base_url:
                    self.clients.append(OpenAI(base_url=self.base_url, api_key="ollama"))
                else:
                    for k in self.api_keys:
                        self.clients.append(OpenAI(api_key=k, base_url=self.base_url))
                self.is_available = len(self.clients) > 0
        except Exception as e:
            self.is_available = False

    def _hash_request(self, messages: List[Dict[str, str]], temperature: float) -> str:
        content_repr = json.dumps({"m": self.model, "msgs": messages, "t": temperature}, sort_keys=True)
        return hashlib.sha256(content_repr.encode("utf-8")).hexdigest()

    def _get_next_client_idx(self) -> Optional[int]:
        with self.lock:
            if not self.clients or self.circuit_open:
                return None
            available_indices = [i for i in range(len(self.clients)) if i not in self.exhausted_keys]
            if not available_indices:
                self.circuit_open = True
                self.is_available = False
                return None
            chosen = self.key_idx % len(self.clients)
            self.key_idx = (self.key_idx + 1) % len(self.clients)
            if chosen in self.exhausted_keys:
                chosen = available_indices[0]
            return chosen

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 500,
        max_retries: int = 6
    ) -> str:
        if not self.is_available or self.circuit_open:
            raise QuotaExhaustedError("LLMClient is not available or circuit breaker is open (quota exhausted).")

        cache_key = self._hash_request(messages, temperature)
        with self.lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        last_err = None
        for attempt in range(max_retries):
            client_idx = self._get_next_client_idx()
            if client_idx is None:
                raise QuotaExhaustedError("All API keys in pool have exhausted quota. Circuit breaker tripped.")

            client = self.clients[client_idx]
            try:
                # Use with_raw_response to inspect live rate-limit headers
                if hasattr(client.chat.completions, "with_raw_response"):
                    raw_resp = client.chat.completions.with_raw_response.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    headers = raw_resp.headers
                    response = raw_resp.parse()
                    
                    # Adaptive rate-limit pacing using Groq's live response headers
                    rem_tokens = headers.get("x-ratelimit-remaining-tokens")
                    if rem_tokens is not None:
                        try:
                            rem_val = int(rem_tokens)
                            if rem_val < 500:
                                reset_str = headers.get("x-ratelimit-reset-tokens", "0.5s").strip().lower()
                                wait_s = 0.5
                                if reset_str.endswith("ms"):
                                    wait_s = float(reset_str[:-2]) / 1000.0
                                elif reset_str.endswith("s"):
                                    wait_s = float(reset_str[:-1])
                                time.sleep(wait_s + 0.25)
                        except Exception:
                            pass
                else:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                choice = response.choices[0].message
                content = choice.content
                if content is not None and content.strip():
                    result = content.strip()
                    with self.lock:
                        self._cache[cache_key] = result
                    return result
                time.sleep(0.5)
            except Exception as e:
                last_err = e
                err_msg = str(e).lower()
                print(f"  [LLMClient Attempt {attempt+1} Exception] {type(e).__name__}: {e}", flush=True)
                
                # Dynamic backoff for provider rate limits (ms, s, m)
                import re
                wait_sec = None
                m = re.search(r"try again in ([0-9mshd\.\s]+)", err_msg)
                if m:
                    dur_str = m.group(1).split(". ")[0].strip().rstrip(".")
                    ms_m = re.match(r"^(\d+(?:\.\d+)?)\s*ms$", dur_str)
                    if ms_m:
                        wait_sec = float(ms_m.group(1)) / 1000.0
                    else:
                        min_sec_m = re.match(r"^(?:(\d+(?:\.\d+)?)\s*m)?\s*(\d+(?:\.\d+)?)\s*s$", dur_str)
                        if min_sec_m:
                            mins = float(min_sec_m.group(1)) if min_sec_m.group(1) else 0.0
                            secs = float(min_sec_m.group(2))
                            wait_sec = mins * 60.0 + secs
                        else:
                            sec_m = re.match(r"^(\d+(?:\.\d+)?)\s*s$", dur_str)
                            if sec_m:
                                wait_sec = float(sec_m.group(1))

                if wait_sec is not None and wait_sec <= 360.0:
                    sleep_dur = max(0.4, wait_sec + 0.5)
                    print(f"  [LLM Rate-Limit Pacing] Sleeping {sleep_dur:.2f}s as instructed by provider...", flush=True)
                    time.sleep(sleep_dur)
                    continue

                is_daily_quota = ("generaterequestsperday" in err_msg or 
                                  "daily quota" in err_msg or
                                  "limit: 500" in err_msg or
                                  "limit: 20" in err_msg or
                                  ("resource_exhausted" in err_msg and "day" in err_msg) or
                                  (wait_sec is not None and wait_sec > 360.0))
                
                if is_daily_quota:
                    with self.lock:
                        self.exhausted_keys.add(client_idx)
                        if len(self.exhausted_keys) >= len(self.clients):
                            self.circuit_open = True
                            self.is_available = False
                    # Fast-fail this key immediately with 0 sleep
                    continue
                else:
                    # Transient error (e.g. 429 RPM or 503 network hiccup), short backoff
                    time.sleep(2.0)

        with self.lock:
            if len(self.exhausted_keys) >= len(self.clients):
                self.circuit_open = True
                self.is_available = False
                raise QuotaExhaustedError(f"API quota exhausted across all available keys. Last error: {last_err}")
                
        raise RuntimeError(f"LLM chat completion failed after {max_retries} attempts. Last error: {last_err}")

