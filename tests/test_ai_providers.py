import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.ai.analyzer import AIAnalyzer
from src.ai.llm_router import LLMRouter, LLMSettings, rank_chat_models, short_error, strip_reasoning
from src.models import IntelligenceItem

_VALID_JSON = (
    '{"summary": "中文摘要", "items": [{"title": "GPT-6", "url": "https://openai.com/index/gpt-6", '
    '"summary": "s", "insight": "i"}], "outlook": "o"}'
)


def _groq_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _gemini_model(name, actions=("generateContent",)):
    return SimpleNamespace(name=f"models/{name}", supported_actions=list(actions))


def _router(*, gemini=None, groq=None, http=None, **settings):
    defaults = {"gemini_model": "gemini-flash-latest", "groq_model": "llama-3.3-70b-versatile"}
    defaults.update(settings)
    return LLMRouter(LLMSettings(**defaults), gemini_client=gemini, groq_client=groq, http=http or MagicMock())


def _news():
    return [IntelligenceItem(title="GPT-6 launch", url="https://openai.com/index/gpt-6", source_type="official_news")]


class TestSettingsFromEnv(unittest.TestCase):
    def test_reads_conventional_names_and_ignores_templates(self):
        settings = LLMSettings.from_env(
            {
                "GEMINI_API_KEY": "{GEMINI_API_KEY}",
                "GROQ_API_KEY": "gsk",
                "NVIDIA_API_KEY": "nvapi",
                "AI_PROVIDER_ORDER": "nvidia, gemini",
                "NVIDIA_ALLOW_PRODUCTION": "true",
                "NVIDIA_BASE_URL": "https://example.test/v1/",
            }
        )
        self.assertIsNone(settings.gemini_api_key)
        self.assertEqual(settings.groq_api_key, "gsk")
        self.assertEqual(settings.provider_order, ("nvidia", "gemini"))
        self.assertTrue(settings.nvidia_enabled)
        self.assertEqual(settings.nvidia_base_url, "https://example.test/v1")
        self.assertEqual(settings.gemini_model, "auto")

    def test_nvidia_disabled_by_default(self):
        self.assertFalse(LLMSettings.from_env({}).nvidia_enabled)


class TestGemini(unittest.TestCase):
    def test_auto_picks_newest_stable_flash(self):
        gemini = MagicMock()
        gemini.models.list.return_value = [
            _gemini_model("gemini-3.7-flash"),
            _gemini_model("gemini-3.8-flash"),
            _gemini_model("gemini-3.9-flash-preview"),
            _gemini_model("gemini-3.8-flash-lite"),
            _gemini_model("gemini-3.8-flash-image"),
            _gemini_model("gemini-3.1-pro"),
            _gemini_model("gemini-omni-1.1-flash"),
            _gemini_model("gemini-4.0-flash", actions=("embedContent",)),
        ]
        router = _router(gemini=gemini, gemini_model="auto")
        self.assertEqual(
            router.gemini_attempt_order(),
            ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.9-flash-preview", "gemini-flash-latest"],
        )

    def test_auto_moves_past_overloaded_model_and_keeps_the_one_that_answered(self):
        gemini = MagicMock()
        gemini.models.list.return_value = [_gemini_model("gemini-3.8-flash"), _gemini_model("gemini-3.7-flash")]
        gemini.models.generate_content.side_effect = [
            Exception("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.'}}"),
            SimpleNamespace(text=_VALID_JSON),
        ]
        router = _router(gemini=gemini, gemini_model="auto")
        result = router.complete("p", json_schema={"type": "object"})
        self.assertTrue(result.ok)
        self.assertEqual((result.provider, result.model), ("gemini", "gemini-3.7-flash"))
        self.assertEqual(result.errors, ["Gemini(gemini-3.8-flash) 503: This model is currently experiencing high demand."])
        gemini.models.generate_content.side_effect = None
        gemini.models.generate_content.return_value = SimpleNamespace(text=_VALID_JSON)
        router.complete("p2")
        self.assertEqual(gemini.models.generate_content.call_args.kwargs["model"], "gemini-3.7-flash")

    def test_auto_falls_back_to_alias_when_listing_fails(self):
        gemini = MagicMock()
        gemini.models.list.side_effect = Exception("boom")
        self.assertEqual(_router(gemini=gemini, gemini_model="auto").gemini_attempt_order(), ["gemini-flash-latest"])

    def test_attempts_are_capped(self):
        gemini = MagicMock()
        gemini.models.list.return_value = [_gemini_model(f"gemini-3.{i}-flash") for i in range(9)]
        router = _router(gemini=gemini, gemini_model="auto", gemini_max_attempts=3)
        self.assertEqual(len(router.gemini_attempt_order()), 3)

    def test_pinned_model_is_not_resolved(self):
        gemini = MagicMock()
        router = _router(gemini=gemini, gemini_model="gemini-custom")
        self.assertEqual(router.gemini_attempt_order(), ["gemini-custom"])
        gemini.models.list.assert_not_called()

    def test_json_schema_requests_json_mime(self):
        gemini = MagicMock()
        gemini.models.generate_content.return_value = SimpleNamespace(text=_VALID_JSON)
        _router(gemini=gemini).complete("p", json_schema={"type": "object", "properties": {"items": {}}})
        config = gemini.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIn("items", config.response_schema["properties"])

    def test_empty_response_counts_as_failure(self):
        gemini = MagicMock()
        gemini.models.generate_content.return_value = SimpleNamespace(text="")
        result = _router(gemini=gemini).complete("p")
        self.assertFalse(result.ok)
        self.assertIn("empty response", result.errors[0])


class TestGroq(unittest.TestCase):
    def test_model_not_found_switches_to_listed_model(self):
        groq = MagicMock()
        groq.chat.completions.create.side_effect = [
            Exception("Error code: 404 - {'error': {'message': 'The model `llama-3.3-70b-versatile` does not exist', 'code': 'model_not_found'}}"),
            _groq_response(_VALID_JSON),
        ]
        groq.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="whisper-large-v3"), SimpleNamespace(id="openai/gpt-oss-120b"), SimpleNamespace(id="llama-4-maverick")]
        )
        router = _router(groq=groq)
        result = router.complete("p")
        self.assertEqual((result.provider, result.model), ("groq", "openai/gpt-oss-120b"))
        self.assertEqual(groq.chat.completions.create.call_args.kwargs["model"], "openai/gpt-oss-120b")

    def test_auto_prefers_family_then_largest(self):
        groq = MagicMock()
        groq.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id=i) for i in ("openai/gpt-oss-20b", "llama-9-400b", "openai/gpt-oss-120b", "whisper-large-v3")]
        )
        groq.chat.completions.create.return_value = _groq_response(_VALID_JSON)
        _router(groq=groq, groq_model="auto").complete("p")
        self.assertEqual(groq.chat.completions.create.call_args.kwargs["model"], "openai/gpt-oss-120b")

    def test_listing_skips_non_chat_models(self):
        groq = MagicMock()
        groq.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id="whisper-large-v3"), SimpleNamespace(id="llama-guard-4")])
        self.assertEqual(_router(groq=groq).groq_candidates(), [])


class TestNvidia(unittest.TestCase):
    def _http(self, content):
        http = MagicMock()
        models_response = MagicMock(status_code=200)
        models_response.json.return_value = {
            "data": [
                {"id": "nvidia/nv-embedqa-e5-v5"},
                {"id": "meta/llama-9-8b-instruct"},
                {"id": "meta/llama-9-405b-instruct"},
                {"id": "nvidia/llama-9-nemotron-guard"},
            ]
        }
        chat_response = MagicMock(status_code=200)
        chat_response.json.return_value = {"choices": [{"message": {"content": content}}]}
        http.get.return_value = models_response
        http.post.return_value = chat_response
        return http

    def test_used_after_gemini_and_groq_fail_when_enabled(self):
        gemini = MagicMock()
        gemini.models.generate_content.side_effect = Exception("503 UNAVAILABLE")
        groq = MagicMock()
        groq.chat.completions.create.side_effect = Exception("Error code: 401 - {'error': {'message': 'Invalid API Key'}}")
        http = self._http("<think>hmm</think>" + _VALID_JSON)
        router = _router(
            gemini=gemini,
            groq=groq,
            http=http,
            nvidia_api_key="nvapi",
            nvidia_enabled=True,
            provider_order=("gemini", "groq", "nvidia"),
        )
        result = router.complete("p")
        self.assertEqual((result.provider, result.model), ("nvidia", "meta/llama-9-405b-instruct"))
        self.assertTrue(result.text.startswith("{"))
        self.assertEqual(len(result.errors), 2)

    def test_not_called_unless_enabled(self):
        http = self._http(_VALID_JSON)
        router = _router(http=http, nvidia_api_key="nvapi", nvidia_enabled=False)
        result = router.complete("p")
        self.assertFalse(result.ok)
        http.post.assert_not_called()
        http.get.assert_not_called()
        self.assertEqual(router.configured_providers(), [])

    def test_provider_order_can_put_nvidia_first(self):
        gemini = MagicMock()
        http = self._http(_VALID_JSON)
        router = _router(
            gemini=gemini, http=http, nvidia_api_key="nvapi", nvidia_enabled=True, provider_order=("nvidia", "gemini")
        )
        self.assertEqual(router.complete("p").provider, "nvidia")
        gemini.models.generate_content.assert_not_called()

    def test_auto_walks_past_models_the_account_cannot_call(self):
        http = self._http(_VALID_JSON)
        not_found = MagicMock(status_code=404, text='{"title":"Not Found","detail":"Function x: Not found for account"}')
        ok = http.post.return_value
        http.post.side_effect = [not_found, ok]
        router = _router(http=http, nvidia_api_key="nvapi", nvidia_enabled=True)
        result = router.complete("p")
        self.assertEqual(result.model, "meta/llama-9-8b-instruct")
        self.assertEqual(len(result.errors), 1)
        self.assertIn("404", result.errors[0])
        router.complete("p2")
        self.assertEqual(http.post.call_args.kwargs["json"]["model"], "meta/llama-9-8b-instruct")

    def test_newer_catalog_entries_first_when_created_is_present(self):
        http = MagicMock()
        listing = MagicMock(status_code=200)
        listing.json.return_value = {
            "data": [
                {"id": "nvidia/llama-3.1-nemotron-70b-instruct", "created": 100},
                {"id": "nvidia/nemotron-3-super-120b", "created": 300},
                {"id": "meta/codellama-70b", "created": 400},
            ]
        }
        http.get.return_value = listing
        router = _router(http=http, nvidia_api_key="nvapi", nvidia_enabled=True)
        self.assertEqual(
            router.nvidia_candidates(), ["nvidia/nemotron-3-super-120b", "nvidia/llama-3.1-nemotron-70b-instruct"]
        )

    def test_http_error_recorded(self):
        http = self._http(_VALID_JSON)
        http.post.return_value = MagicMock(status_code=429, text="Too Many Requests")
        router = _router(http=http, nvidia_api_key="nvapi", nvidia_enabled=True, nvidia_model="meta/x")
        result = router.complete("p")
        self.assertFalse(result.ok)
        self.assertIn("NVIDIA(meta/x) 429", result.errors[0])


class TestDefaults(unittest.TestCase):
    def test_default_order_is_gemini_nvidia_groq(self):
        self.assertEqual(LLMSettings().provider_order, ("gemini", "nvidia", "groq"))
        self.assertEqual(LLMSettings.from_env({}).provider_order, ("gemini", "nvidia", "groq"))


class TestModelSelection(unittest.TestCase):
    def _nvidia_http(self, answers):
        http = MagicMock()
        listing = MagicMock(status_code=200)
        listing.json.return_value = {"data": [{"id": model} for model in answers]}
        http.get.return_value = listing

        def post(url, headers, json, timeout):
            answer = answers[json["model"]]
            if isinstance(answer, int):
                return MagicMock(status_code=answer, text="Not Found")
            response = MagicMock(status_code=200)
            response.json.return_value = {"choices": [{"message": {"content": answer}}]}
            return response

        http.post.side_effect = post
        return http

    def test_picks_best_scoring_model_not_first_ranked(self):
        from src.ai.llm_router import choose_best_model

        http = self._nvidia_http({"llama-9-500b": "poor", "llama-9-120b": "great", "llama-9-70b": 404})
        router = _router(
            http=http, nvidia_api_key="nvapi", nvidia_enabled=True, nvidia_model="auto", probe_model_selection=True
        )
        chosen, log = choose_best_model(
            router, "nvidia", "probe", lambda text: {"great": (1.0,), "poor": (0.3,)}.get(text)
        )
        self.assertEqual(chosen, "llama-9-120b")
        # Best first, the other responsive model kept as fallback, 404 dropped.
        self.assertEqual(router.attempt_order("nvidia"), ["llama-9-120b", "llama-9-500b"])
        self.assertEqual(len(log), 3)

    def test_selection_runs_lazily_once_when_provider_is_reached(self):
        http = self._nvidia_http({"llama-9-500b": "poor", "llama-9-120b": "great"})
        gemini = MagicMock()
        gemini.models.generate_content.return_value = SimpleNamespace(text="from gemini")
        router = _router(
            gemini=gemini, http=http, nvidia_api_key="nvapi", nvidia_enabled=True, probe_model_selection=True
        )
        router.set_model_selector("probe", lambda text: {"great": (1.0,), "poor": (0.3,)}.get(text))
        self.assertEqual(router.complete("p").provider, "gemini")
        http.post.assert_not_called()  # Gemini answered, NVIDIA never probed

        gemini.models.generate_content.side_effect = Exception("503")
        router.complete("p")
        router.complete("p")
        models_called = [call.kwargs["json"]["model"] for call in http.post.call_args_list]
        self.assertEqual(models_called, ["llama-9-500b", "llama-9-120b", "llama-9-120b", "llama-9-120b"])

    def test_pinned_model_skips_selection(self):
        from src.ai.llm_router import choose_best_model

        router = _router(nvidia_api_key="nvapi", nvidia_enabled=True, nvidia_model="meta/x")
        self.assertEqual(choose_best_model(router, "nvidia", "p", lambda t: (1,)), (None, []))


class TestAnalysisQuality(unittest.TestCase):
    def _answer(self, insight, items=3, summary="今日盤勢偏多，半導體領漲。", outlook="留意財報與供應鏈。"):
        import json

        return json.dumps(
            {
                "summary": summary,
                "items": [{"title": f"t{i}", "url": "https://x.test", "summary": "摘要", "insight": insight} for i in range(items)],
                "outlook": outlook,
            },
            ensure_ascii=False,
        )

    def test_orders_answers_by_quality(self):
        from src.ai.analyzer import analysis_quality

        best = analysis_quality(self._answer("這代表需求強勁。後續觀察供應鏈與估值。"), 3)
        shallow = analysis_quality(self._answer("需求強勁"), 3)
        english = analysis_quality(self._answer("Demand is strong. Watch supply.", summary="Bullish.", outlook="Watch."), 3)
        incomplete = analysis_quality(self._answer("這代表需求強勁。後續觀察供應鏈。", items=1), 3)
        self.assertGreater(best, shallow)
        self.assertGreater(best, english)
        self.assertGreater(best, incomplete)
        self.assertEqual(best, (1.0, 2, 1.0, 1.0))

    def test_unusable_answers(self):
        from src.ai.analyzer import analysis_quality

        self.assertIsNone(analysis_quality(None, 3))
        self.assertIsNone(analysis_quality("not json", 3))
        self.assertIsNone(analysis_quality('{"items": [{"title": "x"}]}', 3))


class TestFailover(unittest.TestCase):
    def test_probe_selection_is_off_by_default(self):
        self.assertFalse(LLMSettings().probe_model_selection)
        http = MagicMock()
        router = _router(http=http, nvidia_api_key="nvapi", nvidia_enabled=True)
        router.set_model_selector("probe", lambda text: (1,))
        router._maybe_select("nvidia")
        http.post.assert_not_called()

    def test_groq_walks_to_next_model_on_rate_limit(self):
        groq = MagicMock()
        groq.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="qwen/qwen3.8-27b"), SimpleNamespace(id="openai/gpt-oss-120b")]
        )
        groq.chat.completions.create.side_effect = [
            Exception("Error code: 429 - {'error': {'message': 'Request too large for model on output tokens per minute'}}"),
            _groq_response(_VALID_JSON),
        ]
        router = _router(groq=groq, groq_model="auto")
        result = router.complete("p")
        self.assertTrue(result.ok)
        self.assertEqual(result.provider, "groq")
        self.assertEqual(groq.chat.completions.create.call_count, 2)
        first, second = [call.kwargs["model"] for call in groq.chat.completions.create.call_args_list]
        self.assertNotEqual(first, second)
        self.assertEqual(result.model, second)

    def test_successful_model_is_tried_first_but_others_remain(self):
        http = MagicMock()
        listing = MagicMock(status_code=200)
        listing.json.return_value = {"data": [{"id": "llama-9-500b"}, {"id": "llama-9-120b"}]}
        http.get.return_value = listing
        bad = MagicMock(status_code=503, text="overloaded")
        good = MagicMock(status_code=200)
        good.json.return_value = {"choices": [{"message": {"content": _VALID_JSON}}]}
        http.post.side_effect = [bad, good, bad, good]
        router = _router(http=http, nvidia_api_key="nvapi", nvidia_enabled=True)
        self.assertEqual(router.complete("p").model, "llama-9-120b")
        self.assertEqual(router.attempt_order("nvidia"), ["llama-9-120b", "llama-9-500b"])
        # Next request: the winner fails this time, the other one still gets a try.
        self.assertEqual(router.complete("p2").model, "llama-9-500b")

    def test_retired_and_quota_exhausted_models_are_skipped_for_the_run(self):
        gemini = MagicMock()
        gemini.models.list.return_value = [_gemini_model("gemini-3.8-flash"), _gemini_model("gemini-3.7-flash")]
        gemini.models.generate_content.side_effect = [
            Exception("429 RESOURCE_EXHAUSTED. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
            SimpleNamespace(text=_VALID_JSON),
            SimpleNamespace(text=_VALID_JSON),
        ]
        router = _router(gemini=gemini, gemini_model="auto")
        router.complete("p")
        router.complete("p2")
        models = [call.kwargs["model"] for call in gemini.models.generate_content.call_args_list]
        self.assertEqual(models, ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.7-flash"])

    def test_provider_budget_moves_to_next_provider(self):
        gemini = MagicMock()
        gemini.models.list.return_value = [_gemini_model(f"gemini-3.{i}-flash") for i in range(5)]
        gemini.models.generate_content.side_effect = Exception("503 UNAVAILABLE")
        groq = MagicMock()
        groq.chat.completions.create.return_value = _groq_response(_VALID_JSON)
        router = _router(gemini=gemini, groq=groq, gemini_model="auto", provider_budget_seconds=10)
        # Gemini's first attempt "takes" 100s: its budget is spent, so the
        # chain moves to Groq instead of trying four more Gemini models.
        clock = iter([0, 0, 100, 100, 100, 100, 100, 100])
        with patch("src.ai.llm_router.time.monotonic", side_effect=lambda: next(clock)):
            result = router.complete("p")
        self.assertEqual(result.provider, "groq")
        self.assertEqual(gemini.models.generate_content.call_count, 1)
        self.assertTrue(any("時間上限" in error for error in result.errors))

    def test_long_timeouts_by_default(self):
        settings = LLMSettings()
        self.assertGreaterEqual(settings.nvidia_timeout_seconds, 600)
        self.assertGreaterEqual(settings.provider_budget_seconds, 600)


class TestRouterRobustness(unittest.TestCase):
    def test_no_provider_configured(self):
        result = _router().complete("p")
        self.assertFalse(result.ok)
        self.assertEqual(result.errors, ["沒有可用的 AI 供應商（未設定 API key）"])

    def test_crashing_provider_does_not_stop_the_chain(self):
        gemini = MagicMock()
        groq = MagicMock()
        groq.chat.completions.create.return_value = _groq_response(_VALID_JSON)
        router = _router(gemini=gemini, groq=groq)
        with patch.object(router, "_try_gemini", side_effect=KeyError("bug")):
            result = router.complete("p")
        self.assertEqual(result.provider, "groq")
        self.assertTrue(result.errors[0].startswith("gemini"))

    def test_unknown_provider_in_order_is_skipped(self):
        groq = MagicMock()
        groq.chat.completions.create.return_value = _groq_response(_VALID_JSON)
        self.assertEqual(_router(groq=groq, provider_order=("bogus", "groq")).complete("p").provider, "groq")

    def test_helpers(self):
        self.assertEqual(strip_reasoning("<think>a</think> {}"), "{}")
        self.assertEqual(rank_chat_models(["x-embed", "llama-70b", "llama-8b"], ("llama",), ("embed",)), ["llama-70b", "llama-8b"])
        self.assertEqual(short_error("Gemini", TimeoutError("timed out")), "Gemini: 逾時無回應")


class TestAnalyzerIntegration(unittest.TestCase):
    def _analyzer(self, router):
        analyzer = AIAnalyzer(enable_ai=False)
        analyzer.enable_ai = True
        analyzer.router = router
        return analyzer

    def test_all_providers_failing_marks_report_with_reasons(self):
        gemini = MagicMock()
        gemini.models.generate_content.side_effect = Exception(
            "403 PERMISSION_DENIED. {'error': {'code': 403, 'message': 'Your project has been denied access. Please contact support.'}}"
        )
        groq = MagicMock()
        groq.chat.completions.create.side_effect = Exception("Error code: 401 - {'error': {'message': 'Invalid API Key'}}")
        report = self._analyzer(_router(gemini=gemini, groq=groq)).analyze_ai_tech(_news())
        # Readers only see a plain notice; provider errors stay in metadata/logs.
        self.assertTrue(report.summary.startswith("本輪 AI 分析暫時無法使用"))
        for leaked in ("403", "401", "Gemini", "Groq", "denied", "Invalid API Key"):
            self.assertNotIn(leaked, report.summary)
        self.assertIn("Gemini(gemini-flash-latest) 403: Your project has been denied access", report.metadata["ai_error"])
        self.assertIn("Groq(llama-3.3-70b-versatile) 401: Invalid API Key", report.metadata["ai_error"])

    def test_success_has_no_error_and_records_provider(self):
        gemini = MagicMock()
        gemini.models.generate_content.return_value = SimpleNamespace(text=_VALID_JSON)
        analyzer = self._analyzer(_router(gemini=gemini))
        report = analyzer.analyze_ai_tech(_news())
        self.assertNotIn("ai_error", report.metadata)
        self.assertEqual(report.summary, "中文摘要")
        self.assertEqual(analyzer.last_provider, "gemini:gemini-flash-latest")

    def test_ai_disabled_is_not_flagged_as_failure(self):
        report = AIAnalyzer(enable_ai=False).analyze_ai_tech(_news())
        self.assertNotIn("ai_error", report.metadata)

    def test_analyzer_passes_nvidia_gate_to_router(self):
        self.assertTrue(AIAnalyzer(enable_ai=False, nvidia_enabled=True).router.settings.nvidia_enabled)
        self.assertFalse(AIAnalyzer(enable_ai=False, nvidia_enabled=False).router.settings.nvidia_enabled)

    def test_main_enables_nvidia_only_for_dry_runs_by_default(self):
        import main
        from src.config import Config

        captured = []
        real_analyzer = main.AIAnalyzer

        def recording_analyzer(enable_ai=None, *, nvidia_enabled=None):
            captured.append(nvidia_enabled)
            return real_analyzer(enable_ai=False)

        with patch.object(main, "AIAnalyzer", side_effect=recording_analyzer), patch.object(
            Config, "NVIDIA_ALLOW_PRODUCTION", False
        ), patch.object(Config, "WRITE_ARTIFACTS", False):
            for dry_run in (True, False):
                main.build_reports({}, enable_ai=True, dry_run=dry_run)
        self.assertEqual(captured, [True, False])


if __name__ == "__main__":
    unittest.main()
