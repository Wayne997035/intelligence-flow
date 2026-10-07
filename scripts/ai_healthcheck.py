"""Verify each AI provider answers with real credentials.

Calls every configured provider separately through the same LLMRouter the
analyzer uses and prints one line per provider. Mirrors the analyzer's
fallback chain: passes when at least one provider answers, emits a CI
warning for each provider that fails, and exits 1 only when none works, i.e.
when reports would degrade to local synthesis.

It then scores each provider's candidate models on the production stock and
AI report prompts (sample bundle data) with the same analysis_quality() the
runtime model selection uses, and prints a ranking. Health checks and
evaluation are uses the NVIDIA API Trial Terms allow; NVIDIA calls are spaced
well under the trial rate limit.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.analyzer import AIAnalyzer, analysis_quality  # noqa: E402
from src.ai.llm_router import LLMRouter, LLMSettings  # noqa: E402
from src.config import Config  # noqa: E402
from src.pipeline import normalize_item  # noqa: E402

_PING = json.dumps(
    {"task": "healthcheck", "instructions": ["回傳 JSON：{\"summary\": 用繁體中文一句話說明你是誰}"]},
    ensure_ascii=False,
)
_FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_bundle.json"


def _eval_tasks() -> list[tuple[str, str, int]]:
    """The production stock and AI prompts, filled with the sample bundle."""
    bundle = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    stock_news = [normalize_item(item) for item in bundle.get("stock_news", [])][:4]
    ai_news = [normalize_item(item) for item in bundle.get("ai_news", [])][:4]
    quotes = bundle.get("us_stocks", []) + bundle.get("tw_stocks", [])
    return [
        ("stock", json.dumps(AIAnalyzer.stock_prompt(quotes, stock_news), ensure_ascii=False), min(3, len(stock_news))),
        ("ai", json.dumps(AIAnalyzer.ai_prompt(ai_news), ensure_ascii=False), min(3, len(ai_news))),
    ]


def _settings() -> LLMSettings:
    return LLMSettings(
        gemini_api_key=Config.GEMINI_API_KEY,
        groq_api_key=Config.GROQ_API_KEY,
        nvidia_api_key=Config.NVIDIA_API_KEY,
        gemini_model=Config.AI_MODEL,
        groq_model=Config.GROQ_MODEL,
        nvidia_model=Config.NVIDIA_MODEL,
        nvidia_base_url=Config.NVIDIA_BASE_URL,
        nvidia_enabled=True,  # health check = evaluation, allowed by the trial terms
    )


def _parse(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _check(name: str, settings: LLMSettings) -> bool:
    router = LLMRouter(replace(settings, provider_order=(name,)))
    if name not in router.configured_providers():
        print(f"SKIP {name}: no API key")
        return None
    started = time.monotonic()
    result = router.complete(_PING, json_schema={"type": "object", "properties": {"summary": {"type": "string"}}})
    payload = _parse(result.text)
    elapsed = time.monotonic() - started
    for error in result.errors:
        print(f"  attempt failed: {error}")
    if payload is None:
        print(f"FAIL {name} ({elapsed:.1f}s): {'；'.join(result.errors) or 'response is not JSON'}")
        return False
    print(f"OK   {name} ({result.model}, {elapsed:.1f}s): {str(payload.get('summary', ''))[:100]}")
    return True


def _candidates(settings: LLMSettings, provider: str) -> list[str]:
    router = LLMRouter(settings)
    if provider == "gemini":
        return router.gemini_attempt_order()[:3]
    if provider == "groq":
        return router.groq_candidates()[:4]
    return router.nvidia_candidates()


def _evaluate_models(settings: LLMSettings, responsive_per_provider: int = 5, max_calls_per_provider: int = 16) -> None:
    """Score each provider's candidate models on the real stock and AI
    report prompts with the same analysis_quality() the router uses to pick
    models at runtime. Models the account cannot call (404) are skipped.
    NVIDIA calls are spaced to stay well under its trial rate limit."""
    tasks = _eval_tasks()
    rows = []
    for provider in ("gemini", "groq", "nvidia"):
        if provider not in LLMRouter(replace(settings, provider_order=(provider,))).configured_providers():
            continue
        responsive = calls = 0
        unavailable = []
        for model in _candidates(settings, provider):
            if responsive >= responsive_per_provider or calls >= max_calls_per_provider:
                break
            overrides = {"gemini": "gemini_model", "groq": "groq_model", "nvidia": "nvidia_model"}
            router = LLMRouter(
                replace(
                    settings,
                    provider_order=(provider,),
                    gemini_max_attempts=1,
                    nvidia_timeout_seconds=45,
                    **{overrides[provider]: model},
                )
            )
            scores, seconds, failures = [], 0.0, []
            for name, prompt, expected in tasks:
                started = time.monotonic()
                result = router.complete(prompt, json_schema=AIAnalyzer._REPORT_RESPONSE_SCHEMA if provider == "gemini" else None)
                seconds += time.monotonic() - started
                calls += 1
                if provider == "nvidia":
                    time.sleep(2)
                if not result.ok:
                    failures.extend(result.errors)
                quality = analysis_quality(result.text, expected)
                scores.append(quality)
                if not result.ok and any(" 404" in error for error in result.errors):
                    break
            if failures and all(" 404" in error for error in failures):
                unavailable.append(model)
                continue
            responsive += 1
            usable = [score for score in scores if score is not None]
            combined = tuple(round(sum(values) / len(tasks), 2) for values in zip(*usable)) if len(usable) == len(tasks) else None
            rows.append((combined, -seconds, provider, model, scores, failures))
            detail = f" errors={'；'.join(failures)[:100]}" if failures else ""
            print(f"  eval {provider:6} {model:45} {seconds:6.1f}s stock={scores[0]} ai={scores[-1]}{detail}")
        if unavailable:
            print(f"  eval {provider}: not callable for this account (404): {unavailable}")

    ranked = sorted((row for row in rows if row[0] is not None), key=lambda row: (row[0], row[1]), reverse=True)
    print("  eval ranking (completeness, summary+outlook, insight depth, zh ratio; then speed):")
    for combined, neg_seconds, provider, model, _, _ in ranked:
        print(f"    {combined}  {-neg_seconds:6.1f}s  {provider}:{model}")
    for provider in ("gemini", "groq", "nvidia"):
        best = next((row for row in ranked if row[2] == provider), None)
        if best:
            print(f"  eval best {provider}: {best[3]}")
    if ranked:
        print(f"  eval overall best: {ranked[0][2]}:{ranked[0][3]}")


def main() -> int:
    settings = _settings()
    results = {name: _check(name, settings) for name in ("gemini", "groq", "nvidia")}
    _evaluate_models(settings)

    checked = {name: ok for name, ok in results.items() if ok is not None}
    if not checked:
        print("::error::No AI provider configured; reports will use local synthesis")
        return 1
    for name, ok in checked.items():
        if not ok:
            print(f"::warning::{name} is failing; reports fall back to the next provider")
    if not any(checked.values()):
        print("::error::All AI providers failed; reports will use local synthesis")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
