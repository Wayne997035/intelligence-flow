"""Provider-agnostic LLM routing with automatic model selection and fallback.

Self-contained on purpose so it can be copied into other projects: it only
needs the standard library plus whichever provider SDKs are installed
(``google-genai`` for Gemini, ``groq`` for Groq, ``requests`` for NVIDIA).
It does not import anything from the surrounding application.

Usage::

    router = LLMRouter(LLMSettings.from_env())
    result = router.complete(prompt, json_schema=schema)
    if result.ok:
        use(result.text)          # result.provider / result.model say who answered
    else:
        log(result.errors)        # one short reason per failed attempt

Behaviour:

* Providers are tried in ``provider_order`` (default gemini -> nvidia -> groq);
  the first one that returns text wins. A provider without a key is skipped.
* Model ids default to ``"auto"``: the router asks each provider which models
  the key can call and ranks them generically (Gemini: newest stable Flash
  first; Groq/NVIDIA: preferred family, then largest parameter count), so
  retired or renamed models never need a code change. Any explicit model id
  pins it.
* Gemini in auto mode walks up to ``gemini_max_attempts`` candidates, because
  the newest models are often overloaded (503 / no response) while older
  ones answer. The model that answers is reused for the rest of the process.
* Groq retries once with a listed model when the pinned one is retired.
* NVIDIA in auto mode also walks candidates: its catalog lists models the
  account cannot call (404), which fail fast.
* NVIDIA's free API Catalog is governed by the NVIDIA API Trial Terms
  (trial / development / evaluation only, production use needs a paid
  subscription), so it is only called when ``nvidia_enabled`` is true.
* ``choose_best_model()`` picks the candidate that does best on a probe that
  looks like the caller's real task (quality first, then latency) instead of
  trusting "newest" or "largest".
* Nothing raises: every failure is logged and recorded in ``result.errors``.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - optional dependency
    genai = None
    genai_types = None

try:
    from groq import Groq
except ImportError:  # pragma: no cover - optional dependency
    Groq = None

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None

try:
    from src.utils.redact import redact
except ImportError:  # pragma: no cover - standalone copy of this module
    def redact(text: object) -> str:  # noqa: D103 - minimal fallback
        value = str(text)
        value = re.sub(r"(https?://[^/\s'\"]+)[^\s'\"]*", r"\1/…", value)
        value = re.sub(r"\borg_[A-Za-z0-9]{6,}", "org_***", value)
        return re.sub(r"(account\s*['\"]?)[A-Za-z0-9_\-]{8,}", r"\1***", value, flags=re.I)

PROVIDERS = ("gemini", "groq", "nvidia")
AUTO = "auto"

_GEMINI_FALLBACK_ALIAS = "gemini-flash-latest"
# Variants that are not the general text Flash model.
_GEMINI_SKIP_MARKERS = (
    "lite", "image", "tts", "live", "audio", "embedding", "vision", "native", "robotics", "computer-use",
)
_GROQ_FAMILY_RANK = ("gpt-oss", "llama", "qwen", "kimi", "deepseek", "mistral", "gemma")
_GROQ_NON_CHAT_MARKERS = ("whisper", "guard", "tts", "playai", "embed", "orpheus")
_NVIDIA_FAMILY_RANK = ("gpt-oss", "llama", "nemotron", "qwen", "deepseek", "kimi", "mistral")
_NVIDIA_NON_CHAT_MARKERS = (
    "embed", "rerank", "retriev", "guard", "safety", "reward", "vision", "-vl", "vlm",
    "clip", "parse", "ocr", "deplot", "kosmos", "fuyu", "paligemma", "neva", "cosmos",
    "riva", "asr", "tts", "translate", "coder", "math", "pii", "detector", "classifier",
    "code", "chatqa", "base",
)


def _env(env: dict, name: str, default: str | None = None) -> str | None:
    value = (env.get(name) or "").strip()
    # Treat unfilled templates such as "{GEMINI_API_KEY}" as unset.
    if not value or re.fullmatch(r"\{[A-Z0-9_]+\}", value):
        return default
    return value


@dataclass
class LLMSettings:
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    nvidia_api_key: str | None = None
    gemini_model: str = AUTO
    groq_model: str = AUTO
    nvidia_model: str = AUTO
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    provider_order: tuple[str, ...] = ("gemini", "nvidia", "groq")
    # See module docstring: NVIDIA's free tier is trial / evaluation only.
    nvidia_enabled: bool = False
    # Long reports can take minutes to generate, so a request is given time
    # to finish: providers switch on an explicit failure (429 / 503 / 404 /
    # invalid output), which comes back within seconds, not on a short
    # client timeout. The timeout only guards against a connection that never
    # answers (2026-10-07: a 90s cut-off abandoned requests still being
    # written and every provider ended up failing).
    gemini_timeout_seconds: float = 600
    gemini_max_attempts: int = 5
    groq_timeout_seconds: float = 600
    groq_max_attempts: int = 3
    nvidia_timeout_seconds: float = 600
    # Wall-clock cap per provider per request before moving to the next one.
    provider_budget_seconds: float = 600
    # Probe-based model choice (choose_best_model) before the first real call.
    # Off by default: on free tiers the probes spend rate limits and daily
    # quotas, and a small probe can pick a model that then fails the
    # full-size request (qwen on Groq, 2026-10-07). Ranked order + failover
    # is used instead; the AI health check evaluates models on real prompts.
    probe_model_selection: bool = False
    # The NVIDIA catalog lists many models the account cannot call (404
    # "Function ... Not found for account"); those fail in ~0.1s, so auto
    # mode walks this many candidates before giving up.
    nvidia_max_attempts: int = 8

    @classmethod
    def from_env(cls, env: dict | None = None, **overrides: Any) -> "LLMSettings":
        """Read the conventional variable names: GEMINI_API_KEY, GROQ_API_KEY,
        NVIDIA_API_KEY, AI_MODEL (Gemini), GROQ_MODEL, NVIDIA_MODEL,
        NVIDIA_BASE_URL, AI_PROVIDER_ORDER, NVIDIA_ALLOW_PRODUCTION."""
        env = os.environ if env is None else env
        order = _env(env, "AI_PROVIDER_ORDER")
        settings = cls(
            gemini_api_key=_env(env, "GEMINI_API_KEY"),
            groq_api_key=_env(env, "GROQ_API_KEY"),
            nvidia_api_key=_env(env, "NVIDIA_API_KEY"),
            gemini_model=_env(env, "AI_MODEL", AUTO) or AUTO,
            groq_model=_env(env, "GROQ_MODEL", AUTO) or AUTO,
            nvidia_model=_env(env, "NVIDIA_MODEL", AUTO) or AUTO,
            nvidia_base_url=(_env(env, "NVIDIA_BASE_URL", cls.nvidia_base_url) or cls.nvidia_base_url).rstrip("/"),
            provider_order=tuple(p.strip().lower() for p in order.split(",") if p.strip()) if order else cls.provider_order,
            nvidia_enabled=(_env(env, "NVIDIA_ALLOW_PRODUCTION", "false") or "").lower() in {"1", "true", "yes", "on"},
        )
        for key, value in overrides.items():
            setattr(settings, key, value)
        return settings


@dataclass
class LLMResult:
    text: str | None = None
    provider: str | None = None
    model: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.text)


class LLMRouter:
    def __init__(
        self,
        settings: LLMSettings,
        *,
        gemini_client: Any = None,
        groq_client: Any = None,
        http: Any = None,
        logger: logging.Logger | None = None,
    ):
        self.settings = settings
        self.log = logger or logging.getLogger(__name__)
        self.http = http or requests
        self.gemini_client = gemini_client if gemini_client is not None else self._make_gemini_client()
        self.groq_client = groq_client if groq_client is not None else self._make_groq_client()
        # Selected model per provider; "auto" until resolved.
        self.models = {
            "gemini": settings.gemini_model or AUTO,
            "groq": settings.groq_model or AUTO,
            "nvidia": settings.nvidia_model or AUTO,
        }
        # Ranked candidates per provider (resolved lazily), models that cannot
        # be called this run, and the model that last answered per provider.
        self._candidates: dict[str, list[str] | None] = {name: None for name in PROVIDERS}
        self._dead: set[tuple[str, str]] = set()
        self.last_model: dict[str, str] = {}
        # Optional task-specific probe used to pick models (see
        # set_model_selector); runs at most once per provider, and only when
        # the chain actually reaches that provider.
        self._selector: tuple[str, Callable[[str], tuple | None]] | None = None
        self._selection_done: set[str] = set()
        self.selection_log: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ setup
    def _make_gemini_client(self):
        if not (genai and self.settings.gemini_api_key):
            return None
        try:
            options = (
                genai_types.HttpOptions(timeout=int(self.settings.gemini_timeout_seconds * 1000))
                if genai_types
                else None
            )
            return genai.Client(api_key=self.settings.gemini_api_key, http_options=options)
        except Exception as exc:  # pragma: no cover - SDK construction failure
            self.log.warning("Gemini client setup failed: %s", exc)
            return None

    def _make_groq_client(self):
        if not (Groq and self.settings.groq_api_key):
            return None
        try:
            # The router does its own failover; the SDK's built-in retries waited
            # 47s on a 429 that would never succeed (2026-10-07).
            return Groq(api_key=self.settings.groq_api_key, timeout=self.settings.groq_timeout_seconds, max_retries=0)
        except Exception as exc:  # pragma: no cover - SDK construction failure
            self.log.warning("Groq client setup failed: %s", exc)
            return None

    def set_model_selector(self, probe_prompt: str, score: Callable[[str], tuple | None]) -> None:
        """Enable probe-based model choice for providers left on "auto"
        (Groq, NVIDIA). Gemini keeps newest-first with failover since its
        candidates are versions of the same Flash model."""
        self._selector = (probe_prompt, score)

    def _maybe_select(self, provider: str) -> None:
        if not self.settings.probe_model_selection:
            return
        if self._selector is None or provider in self._selection_done:
            return
        if self.models.get(provider, AUTO).lower() != AUTO:
            return
        self._selection_done.add(provider)
        probe, score = self._selector
        chosen, log = choose_best_model(
            self, provider, probe, score, pause_seconds=1.0 if provider == "nvidia" else 0.0
        )
        self.selection_log[provider] = log
        if chosen is None:
            self.log.warning("%s model probe found no suitable model; using ranked order.", provider)

    def configured_providers(self) -> list[str]:
        """Providers that will actually be tried, in order."""
        available = {
            "gemini": self.gemini_client is not None,
            "groq": self.groq_client is not None,
            "nvidia": bool(self.settings.nvidia_api_key and self.settings.nvidia_enabled and self.http),
        }
        return [name for name in self.settings.provider_order if available.get(name)]

    # --------------------------------------------------------------- routing
    def complete(self, prompt: str, *, json_schema: dict | None = None) -> LLMResult:
        result = LLMResult()
        if not self.configured_providers():
            result.errors.append("沒有可用的 AI 供應商（未設定 API key）")
            return result
        for provider in self.settings.provider_order:
            name = provider.strip().lower()
            if name not in PROVIDERS:
                self.log.warning("Unknown AI provider %r in provider order; skipping.", provider)
                continue
            text = self.try_provider(name, prompt, json_schema=json_schema, errors=result.errors)
            if text:
                result.text, result.provider, result.model = text, name, self.last_model.get(name, self.models[name])
                return result
        return result

    def try_provider(
        self, name: str, prompt: str, *, json_schema: dict | None = None, errors: list[str] | None = None
    ) -> str | None:
        errors = errors if errors is not None else []
        try:
            if name == "gemini":
                return self._try_gemini(prompt, json_schema, errors)
            if name == "groq":
                return self._try_groq(prompt, errors)
            if name == "nvidia":
                return self._try_nvidia(prompt, errors)
        except Exception as exc:  # defensive: a provider bug must not stop the chain
            errors.append(short_error(name, exc))
            self.log.exception("AI provider %s crashed", name)
        return None

    # ------------------------------------------------------- candidate walk
    def _walk(self, provider: str, call: Callable[[str], str | None], errors: list[str]) -> str | None:
        """Try this provider's candidate models in order until one answers.

        Every failure (rate limit, overload, timeout, retired model, bad
        output) moves on to the next candidate; the provider as a whole gets
        `provider_budget_seconds`, after which the chain moves to the next
        provider instead of waiting out more timeouts. Models that cannot be
        called at all (404 / exhausted daily quota) are skipped for the rest
        of the process; the model that answers is tried first next time.
        """
        label = _LABELS[provider]
        deadline = time.monotonic() + self.settings.provider_budget_seconds
        cap = max(1, _MAX_ATTEMPTS[provider](self.settings))
        tried: set[str] = set()
        while len(tried) < cap:
            # Re-read the order each time: a pinned model that turns out to be
            # retired falls back to the auto-ranked candidates.
            remaining = [model for model in self.attempt_order(provider) if model not in tried]
            if not remaining:
                break
            if time.monotonic() >= deadline:
                errors.append(f"{label}: 超過 {self.settings.provider_budget_seconds:.0f} 秒時間上限，換下一家")
                self.log.warning("%s exceeded its time budget; moving to the next provider.", label)
                break
            model = remaining[0]
            tried.add(model)
            try:
                text = call(model)
                if not text or not text.strip():
                    raise RuntimeError("empty response")
            except Exception as exc:
                self.log.warning("%s %s failed: %s", label, model, exc)
                errors.append(short_error(f"{label}({model})", exc))
                if _is_unusable_for_run(exc):
                    self._dead.add((provider, model))
                continue
            self._promote(provider, model)
            self.last_model[provider] = model
            return text
        return None

    def attempt_order(self, provider: str) -> list[str]:
        pinned = self.models[provider]
        if pinned.lower() != AUTO and (provider, pinned) not in self._dead:
            return [pinned]
        if self._candidates.get(provider) is None:
            ranked = {"gemini": self._gemini_ranked, "groq": self.groq_candidates, "nvidia": self.nvidia_candidates}[
                provider
            ]()
            self._candidates[provider] = ranked
            self.log.info("%s candidates: %s", _LABELS[provider], ", ".join(ranked[:12]))
        usable = [model for model in self._candidates[provider] if (provider, model) not in self._dead]
        return usable[: max(1, _MAX_ATTEMPTS[provider](self.settings))]

    def _promote(self, provider: str, model: str) -> None:
        candidates = self._candidates.get(provider)
        if candidates and model in candidates and candidates[0] != model:
            candidates.remove(model)
            candidates.insert(0, model)
            self.log.info("%s model in use: %s", _LABELS[provider], model)

    # ---------------------------------------------------------------- gemini
    def _try_gemini(self, prompt: str, json_schema: dict | None, errors: list[str]) -> str | None:
        if self.gemini_client is None:
            return None

        def call(model: str) -> str | None:
            config = None
            if genai_types is not None:
                config = genai_types.GenerateContentConfig(
                    response_mime_type="application/json" if json_schema else None,
                    response_schema=json_schema,
                )
            response = self.gemini_client.models.generate_content(model=model, contents=prompt, config=config)
            return getattr(response, "text", None)

        return self._walk("gemini", call, errors)

    def gemini_attempt_order(self) -> list[str]:
        return self.attempt_order("gemini")

    def _gemini_ranked(self) -> list[str]:
        ranked = self.gemini_candidates()
        if _GEMINI_FALLBACK_ALIAS not in ranked:
            ranked.append(_GEMINI_FALLBACK_ALIAS)
        return ranked

    def gemini_candidates(self) -> list[str]:
        """General Flash models this key can call: newest stable first, then
        previews. Empty when listing fails (the alias is used instead)."""
        if self.gemini_client is None:
            return []
        try:
            models = list(self.gemini_client.models.list())
        except Exception as exc:
            self.log.warning("Gemini model listing failed: %s", exc)
            return []
        ranked: list[tuple[tuple, str]] = []
        for model in models:
            name = (getattr(model, "name", "") or "").removeprefix("models/")
            lowered = name.lower()
            if "generateContent" not in (getattr(model, "supported_actions", None) or []) or "flash" not in lowered:
                continue
            if any(marker in lowered for marker in _GEMINI_SKIP_MARKERS):
                continue
            version = re.search(r"gemini-(\d+(?:\.\d+)?)", lowered)
            if not version:
                continue
            unstable = any(tag in lowered for tag in ("preview", "exp"))
            ranked.append(((unstable, -float(version.group(1)), len(name)), name))
        return [name for _, name in sorted(ranked)]

    # ------------------------------------------------------------------ groq
    def _try_groq(self, prompt: str, errors: list[str]) -> str | None:
        if self.groq_client is None:
            return None
        self._maybe_select("groq")

        def call(model: str) -> str | None:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                # Without JSON mode models often prefix prose around the JSON.
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        return self._walk("groq", call, errors)

    def groq_candidates(self) -> list[str]:
        if self.groq_client is None:
            return []
        try:
            data = getattr(self.groq_client.models.list(), "data", []) or []
        except Exception as exc:
            self.log.warning("Groq model listing failed: %s", exc)
            return []
        ids = [getattr(model, "id", None) for model in data]
        return rank_chat_models([i for i in ids if i], _GROQ_FAMILY_RANK, _GROQ_NON_CHAT_MARKERS)

    # ---------------------------------------------------------------- nvidia
    def _try_nvidia(self, prompt: str, errors: list[str]) -> str | None:
        if not self.settings.nvidia_api_key:
            return None
        if not self.settings.nvidia_enabled:
            self.log.info("NVIDIA skipped: free API Catalog is trial/evaluation-only (nvidia_enabled is false).")
            return None
        self._maybe_select("nvidia")

        def call(model: str) -> str | None:
            response = self.http.post(
                f"{self.settings.nvidia_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.nvidia_api_key}", "Accept": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 4096,
                },
                timeout=self.settings.nvidia_timeout_seconds,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"{response.status_code} {response.text[:200]}")
            return strip_reasoning(response.json()["choices"][0]["message"]["content"])

        return self._walk("nvidia", call, errors)

    def nvidia_attempt_order(self) -> list[str]:
        return self.attempt_order("nvidia")

    def nvidia_candidates(self) -> list[str]:
        if not (self.settings.nvidia_api_key and self.http):
            return []
        try:
            response = self.http.get(
                f"{self.settings.nvidia_base_url}/models",
                headers={"Authorization": f"Bearer {self.settings.nvidia_api_key}"},
                timeout=20,
            )
            response.raise_for_status()
            entries = [entry for entry in response.json().get("data", []) if entry.get("id")]
        except Exception as exc:
            self.log.warning("NVIDIA model listing failed: %s", exc)
            return []
        ranked = rank_chat_models([entry["id"] for entry in entries], _NVIDIA_FAMILY_RANK, _NVIDIA_NON_CHAT_MARKERS)
        created = {entry["id"]: entry.get("created") for entry in entries}
        if len({created.get(model_id) for model_id in ranked if isinstance(created.get(model_id), (int, float))}) > 1:
            # Newer catalog entries first when the listing carries real
            # creation times; names alone cannot tell generations apart
            # ("llama-3.1-nemotron" vs "nemotron-3"). Stable sort keeps the
            # family/size order as the tie-breaker.
            ranked.sort(key=lambda model_id: -(created.get(model_id) or 0))
        return ranked


_LABELS = {"gemini": "Gemini", "groq": "Groq", "nvidia": "NVIDIA"}
_MAX_ATTEMPTS = {
    "gemini": lambda settings: settings.gemini_max_attempts,
    "groq": lambda settings: settings.groq_max_attempts,
    "nvidia": lambda settings: settings.nvidia_max_attempts,
}


def _is_unusable_for_run(exc: Exception) -> bool:
    """Errors that will not clear within this run: the model does not exist
    for this account or its daily free quota is used up."""
    text = str(exc)
    lowered = text.lower()
    return (
        "404" in text
        or "model_not_found" in lowered
        or "no longer available" in lowered
        or ("resource_exhausted" in lowered and "per day" in lowered.replace("perday", "per day"))
        or "requestsperday" in lowered
    )


# --------------------------------------------------------- model selection
def choose_best_model(
    router: "LLMRouter",
    provider: str,
    probe_prompt: str,
    score: Callable[[str], tuple | None],
    *,
    max_responsive: int = 4,
    max_calls: int = 10,
    probe_timeout_seconds: float = 30,
    pause_seconds: float = 0.0,
) -> tuple[str | None, list[str]]:
    """Pick the candidate model that does best on a task-specific probe.

    "Newest" or "largest" is not "most suitable" (on 2026-10-07 the 550B
    NVIDIA model wrote the least Chinese of the working candidates), so the
    caller supplies a probe prompt that looks like the real task and a
    `score(text)` returning a comparable tuple (higher is better) or None for
    an unusable answer. Models that are not callable (404) fail fast and do
    not count toward `max_responsive`. Latency is the final tie-breaker.
    On success the router is switched to the chosen model and the decision
    log is returned for observability.
    """
    log: list[str] = []
    candidates = {"groq": router.groq_candidates, "nvidia": router.nvidia_candidates}.get(provider)
    if candidates is None or router.models.get(provider, AUTO).lower() != AUTO:
        return None, log
    original_timeout = router.settings.nvidia_timeout_seconds
    router.settings.nvidia_timeout_seconds = probe_timeout_seconds
    scored: list[tuple[tuple, str]] = []
    unusable: list[str] = []
    responsive = calls = 0
    try:
        for model in candidates():
            if responsive >= max_responsive or calls >= max_calls:
                break
            router.models[provider] = model
            errors: list[str] = []
            started = time.monotonic()
            text = router.try_provider(provider, probe_prompt, errors=errors)
            elapsed = time.monotonic() - started
            calls += 1
            if pause_seconds:
                time.sleep(pause_seconds)
            if not text:
                log.append(f"{model}: {'；'.join(errors) or 'no answer'}")
                if any(" 404" in error for error in errors):
                    unusable.append(model)
                else:
                    responsive += 1
                continue
            responsive += 1
            quality = score(text)
            log.append(f"{model}: {elapsed:.1f}s score={quality}")
            if quality is not None:
                scored.append(((tuple(quality), -elapsed), model))
    finally:
        router.settings.nvidia_timeout_seconds = original_timeout
        router.models[provider] = AUTO
    if not scored:
        return None, log
    ranked = [model for _, model in sorted(scored, reverse=True)]
    # Best-scoring models first, then the rest in their original order so the
    # walk still has fallbacks when the chosen model fails a full request.
    rest = [model for model in (router._candidates.get(provider) or candidates()) if model not in ranked]
    router._candidates[provider] = ranked + [model for model in rest if model not in unusable]
    best = (None, ranked[0])
    router.log.info("%s model chosen by probe: %s", provider, best[1])
    return best[1], log


# ------------------------------------------------------------------ helpers
def rank_chat_models(ids: list[str], families: tuple[str, ...], skip_markers: tuple[str, ...]) -> list[str]:
    """Drop non-chat models, then order by family rank and largest size.

    Matching is on the id's model part (after "org/"), so an org name such as
    "nvidia/" or "meta/" does not decide the family."""
    usable = [
        model_id
        for model_id in ids
        if not any(marker in model_id.lower().rsplit("/", 1)[-1] for marker in skip_markers)
    ]

    def rank(model_id: str) -> tuple:
        lowered = model_id.lower().rsplit("/", 1)[-1]
        family = next((index for index, name in enumerate(families) if name in lowered), len(families))
        sizes = [float(size) for size in re.findall(r"(\d+(?:\.\d+)?)b\b", lowered)]
        return (family, -(max(sizes) if sizes else 0.0), model_id)

    return sorted(usable, key=rank)


def strip_reasoning(content: str | None) -> str | None:
    """Remove <think>…</think> blocks some reasoning models prepend."""
    if not content:
        return content
    return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()


def short_error(provider: str, exc: Exception) -> str:
    text = redact(str(exc) or type(exc).__name__)
    if "timeout" in type(exc).__name__.lower() or "timed out" in text.lower():
        return f"{provider}: 逾時無回應"
    match = re.search(r"'message':\s*'([^']+)'", text)
    detail = match.group(1) if match else text
    code = re.search(r"\b(4\d\d|5\d\d)\b", text)
    prefix = f"{provider} {code.group(1)}" if code else provider
    return f"{prefix}: {detail[:120]}"


