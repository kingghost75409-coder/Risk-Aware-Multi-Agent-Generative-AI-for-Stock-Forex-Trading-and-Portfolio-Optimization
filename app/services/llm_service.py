"""Unified LLM gateway — supports Google Gemini, Anthropic Claude, Local Ollama, and smart fallbacks.

Provider priority (uses first available):
1. Google Gemini (GEMINI_API_KEY) — Free tier available via Google AI Studio
2. Anthropic Claude (ANTHROPIC_API_KEY)
3. Local Ollama (OLLAMA_URL)
4. Enhanced Financial Explainer Fallback (Structured, explainable templates)

The LLM produces humanized, consultative, deeply explainable financial analysis,
translating complex quant indicators, agent consensus, and risk metrics into plain English.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an empathetic, world-class financial mentor and senior quantitative trading assistant for Stock AI Trader.
Your mission is to make stock trading, multi-agent AI analysis, and risk management completely understandable, explainable, and helpful for users of all experience levels—especially beginners.

COMMUNICATION PRINCIPLES:
1. HUMANIZED, CONVERSATIONAL & EMPATHETIC:
   - Speak like a friendly, patient, experienced Wall Street mentor sitting right next to the user.
   - Never sound like a rigid robot regurgitating the exact same bulleted template.
   - If the user greets you (e.g. "hi", "hello"), respond naturally and warmly, introducing how you can help them navigate the markets and terminal.
   - Acknowledge beginner questions with encouragement and simplify complex Wall Street jargon into clear analogies.
2. RIGOROUS YET ACCESSIBLE:
   - When discussing stocks or signals, explain the "WHY" in simple terms (e.g., what RSI means, why the ML model is cautious, why the risk engine recommends a small position size).
   - Tailor the answer to the user's specific question rather than giving unrelated information.
   - Emphasize that all insights are for educational and paper-trading simulation purposes.
3. FACTUAL FIDELITY:
   - Ground factual data (prices, signals, confidence, portfolio balances) strictly in the live context provided. Never hallucinate fake numbers.
"""


class LLMResponse:
    """Normalized response from any LLM provider."""

    def __init__(
        self,
        text: str,
        provider: str,
        model: str,
        tokens_used: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        correlation_id: str | None = None,
        structured_data: dict[str, Any] | None = None,
    ):
        self.text = text
        self.provider = provider
        self.model = model
        self.tokens_used = tokens_used
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_ms = latency_ms
        self.correlation_id = correlation_id
        self.structured_data = structured_data


class OpenRouterProvider:
    """OpenRouter API Provider — OpenAI-compatible Chat Completions interface.

    Official API documentation: https://openrouter.ai/docs
    Responsibilities:
    - Model routing and OpenAI-compatible completions
    - Bounded exponential backoff with configurable retries
    - Token accounting and latency tracking
    - Safe error logging without secret exposure
    - Structured JSON output parsing when requested
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        app_name: str | None = None,
        site_url: str | None = None,
        timeout: int = 25,
        max_retries: int = 3,
    ):
        self.api_key = api_key.strip()
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.app_name = app_name or os.getenv("OPENROUTER_APP_NAME", "AI Investment Intelligence")
        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL", "https://stock-ai-trader.local")
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
        system: str = SYSTEM_PROMPT,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.35,
        max_tokens: int = 1500,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse | None:
        import urllib.error
        import urllib.request

        correlation_id = str(uuid.uuid4())[:8]
        endpoint = f"{self.base_url}/chat/completions"

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            for msg in history[-6:]:
                role = "user" if msg.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
            "Content-Type": "application/json",
            "User-Agent": f"StockAITrader/{self.app_name}",
        }

        data = json.dumps(payload).encode("utf-8")
        delay = 0.5

        for attempt in range(1, self.max_retries + 1):
            start_t = time.perf_counter()
            try:
                req = urllib.request.Request(endpoint, data=data, method="POST", headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw_body = resp.read().decode("utf-8")
                    result = json.loads(raw_body)

                latency_ms = (time.perf_counter() - start_t) * 1000.0
                choices = result.get("choices", [])
                if choices and "message" in choices[0]:
                    text = choices[0]["message"].get("content", "")
                    usage = result.get("usage", {})
                    p_tok = usage.get("prompt_tokens", 0)
                    c_tok = usage.get("completion_tokens", 0)
                    tot_tok = usage.get("total_tokens", p_tok + c_tok)

                    structured_data = None
                    if response_format and text:
                        try:
                            structured_data = json.loads(text)
                        except Exception:
                            logger.debug("Response text was not parseable as structured JSON")

                    return LLMResponse(
                        text=text,
                        provider="openrouter",
                        model=self.model,
                        tokens_used=tot_tok,
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        latency_ms=latency_ms,
                        correlation_id=correlation_id,
                        structured_data=structured_data,
                    )
            except urllib.error.HTTPError as http_err:
                latency_ms = (time.perf_counter() - start_t) * 1000.0
                err_msg = ""
                try:
                    err_body = http_err.read().decode("utf-8", errors="ignore")
                    err_msg = json.loads(err_body).get("error", {}).get("message", err_body[:200])
                except Exception:
                    err_msg = str(http_err)
                logger.warning(
                    "OpenRouter HTTP %d on attempt %d/%d (cid=%s, latency=%.1fms): %s",
                    http_err.code, attempt, self.max_retries, correlation_id, latency_ms, err_msg
                )
                if http_err.code == 404 and self.model != "openai/gpt-4o-mini":
                    logger.warning("Model '%s' retired/unavailable on OpenRouter (404). Auto-falling back to 'openai/gpt-4o-mini'.", self.model)
                    self.model = "openai/gpt-4o-mini"
                    payload["model"] = "openai/gpt-4o-mini"
                    data = json.dumps(payload).encode("utf-8")
                    continue
                if http_err.code in (401, 403):
                    # Permanent auth error - do not retry
                    break
            except Exception as exc:
                latency_ms = (time.perf_counter() - start_t) * 1000.0
                logger.warning(
                    "OpenRouter call failed on attempt %d/%d (cid=%s, latency=%.1fms): %s",
                    attempt, self.max_retries, correlation_id, latency_ms, exc
                )

            if attempt < self.max_retries:
                time.sleep(delay)
                delay = min(delay * 2.0, 2.0)

        return None


class GeminiProvider:
    """Google Gemini API Provider with model fallback and multi-turn context."""

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT,
                 history: list[dict[str, str]] | None = None) -> LLMResponse | None:
        import urllib.request

        models_to_try = list(dict.fromkeys([
            self.model,
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
            "gemini-flash-latest",
            "gemini-3.5-flash",
        ]))

        contents = []
        if history:
            for msg in history[-6:]:
                role = "user" if msg.get("role") == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            payload = {
                "contents": contents,
                "systemInstruction": {"parts": [{"text": system}]},
                "generationConfig": {
                    "temperature": 0.35,
                    "maxOutputTokens": 1500,
                },
            }
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url, data=data, method="POST",
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=25) as resp:
                    result = json.loads(resp.read().decode("utf-8"))

                candidates = result.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        text = parts[0]["text"]
                        tokens = result.get("usageMetadata", {}).get("totalTokenCount", 0)
                        return LLMResponse(text=text, provider="gemini", model=model, tokens_used=tokens)
            except Exception as exc:
                logger.warning("Gemini API call failed for model %s: %s", model, exc)
                continue

        return None


class AnthropicProvider:
    """Anthropic Claude API Provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self.model = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT,
                 history: list[dict[str, str]] | None = None) -> LLMResponse | None:
        import urllib.request

        url = "https://api.anthropic.com/v1/messages"
        messages = []
        if history:
            for msg in history[-6:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "max_tokens": 1500,
            "system": system,
            "messages": messages,
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST", headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            text = result["content"][0]["text"]
            tokens = result.get("usage", {}).get("input_tokens", 0) + result.get("usage", {}).get("output_tokens", 0)
            return LLMResponse(text=text, provider="anthropic", model=self.model, tokens_used=tokens)
        except Exception as exc:
            logger.warning("Anthropic API failed: %s", exc)
            return None


class OllamaProvider:
    """Local Ollama server provider."""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2")

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT,
                 history: list[dict[str, str]] | None = None) -> LLMResponse | None:
        import urllib.request

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.35, "num_predict": 1024},
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return LLMResponse(text=result.get("response", ""), provider="ollama", model=self.model)
        except Exception as exc:
            logger.debug("Ollama generation failed (server likely offline): %s", exc)
            return None


class TemplateFallback:
    """Smart structured fallback when no live LLM connection is available."""

    def format_analysis(self, tool_output: str) -> LLMResponse:
        return LLMResponse(text=tool_output, provider="template", model="none")


class LLMGateway:
    """Unified interface — tries live LLMs in order, falls back gracefully."""

    def __init__(self):
        self.providers: list = []
        self.fallback = TemplateFallback()
        self.reload()

    def reload(self) -> None:
        """Re-check environment and instantiate active providers in priority order."""
        self.providers = []

        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

        # Intelligent key detection: automatically resolve misplaced keys
        if gemini_key.startswith("sk-or-") and not openrouter_key:
            logger.info("Auto-detect: Moving OpenRouter key from GEMINI_API_KEY to OpenRouterProvider")
            openrouter_key = gemini_key
            gemini_key = ""
        elif openrouter_key.startswith("AIzaSy") and not gemini_key:
            logger.info("Auto-detect: Moving Google Gemini key from OPENROUTER_API_KEY to GeminiProvider")
            gemini_key = openrouter_key
            openrouter_key = ""

        # 1. OpenRouter (primary unified model provider if key configured)
        if openrouter_key:
            provider = OpenRouterProvider(openrouter_key)
            self.providers.append(provider)
            logger.info("LLM: OpenRouter provider registered (%s)", provider.model)

        # 2. Google Gemini
        if gemini_key:
            self.providers.append(GeminiProvider(gemini_key))
            logger.info("LLM: Gemini provider registered (%s)", os.getenv("GEMINI_MODEL", "gemini-3.6-flash"))

        # 3. Anthropic Claude
        if anthropic_key:
            self.providers.append(AnthropicProvider(anthropic_key))
            logger.info("LLM: Anthropic provider registered")

        # 4. Local Ollama
        self.providers.append(OllamaProvider())

    @property
    def is_llm_active(self) -> bool:
        """Returns True if a cloud API or working local model is configured."""
        for p in self.providers:
            if isinstance(p, (OpenRouterProvider, GeminiProvider, AnthropicProvider)):
                return True
        return False

    @property
    def active_provider_name(self) -> str:
        """Name of the currently configured primary provider."""
        for p in self.providers:
            if isinstance(p, OpenRouterProvider):
                return f"OpenRouter ({p.model})"
            if isinstance(p, GeminiProvider):
                return f"Gemini ({p.model})"
            if isinstance(p, AnthropicProvider):
                return f"Claude ({p.model})"
        return "Template (No Key)"

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT,
                 history: list[dict[str, str]] | None = None) -> LLMResponse:
        """Generate response through first successful provider."""
        for provider in self.providers:
            result = provider.generate(prompt, system, history=history)
            if result and result.text:
                return result
        return self.fallback.format_analysis(prompt)

    def generate_conversational(self, question: str, history: list[dict[str, str]] | None = None,
                                stock_context: str = "") -> LLMResponse | None:
        """Generate a natural conversational response for open-ended financial questions."""
        ctx_block = f"Live Terminal Context:\n{stock_context}" if stock_context else ""
        prompt = f"""User Question: "{question}"

{ctx_block}

Please provide an insightful, encouraging, and human-centered response.
- If the user greets you or asks for general advice, respond warmly and conversationally as their trading mentor.
- If the user mentions being a beginner or having less knowledge, explain any trading concepts in simple, intuitive, step-by-step terms with everyday analogies.
- Directly address what the user actually asked.
- Use clean formatting (bullet points, bold highlights) to keep the text engaging and readable.
- State that this is for educational and simulated paper-trading purposes."""
        for provider in self.providers:
            res = provider.generate(prompt, system=SYSTEM_PROMPT, history=history)
            if res and res.text:
                return res
        return None

    def format_tool_output(self, question: str, tool_name: str,
                            raw_output: str, context: str = "",
                            history: list[dict[str, str]] | None = None) -> LLMResponse:
        """Format a tool's raw output into natural, humanized language."""
        ctx_block = f"Recent Conversation History:\n{context}" if context else ""
        prompt = f"""The user asked: "{question}"

Tool Executed: {tool_name}
Verified Quantitative Data:
{raw_output}

{ctx_block}

Please synthesize this into a clear, explainable, humanized response addressing the user's question.
- Speak conversationally and empathetically like a trusted financial advisor and mentor.
- Answer the user's specific question directly based on the verified data.
- If the user asks for analysis or a buy/sell question, explain the underlying logic (what the technical momentum, ML forecast, and risk guardrails see) in simple English.
- Avoid robotic or repetitive phrasing. Never invent numbers."""
        return self.generate(prompt, history=history)


# Singleton gateway
_gateway: LLMGateway | None = None


def get_llm_gateway() -> LLMGateway:
    """Get or create the singleton LLM gateway."""
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway


def reload_llm_gateway() -> LLMGateway:
    """Force reload the singleton LLM gateway with updated environment variables."""
    global _gateway
    _gateway = LLMGateway()
    return _gateway
