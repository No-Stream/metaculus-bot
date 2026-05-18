from metaculus_bot.api_key_utils import get_openrouter_api_key


class TestApiKeyUtils:
    def test_openai_model_uses_special_key(self, monkeypatch):
        """OpenAI models should use OAI_ANTH_OPENROUTER_KEY when available."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("openrouter/openai/gpt-5")
        assert result == "special_key"

    def test_anthropic_model_uses_special_key(self, monkeypatch):
        """Anthropic models should use OAI_ANTH_OPENROUTER_KEY when available."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("openrouter/anthropic/claude-sonnet-4")
        assert result == "special_key"

    def test_google_model_uses_special_key_when_donated_toggle_on(self, monkeypatch):
        """Google models route via the donated key when GEMINI_USE_DONATED_OPENROUTER_KEY=true.

        Added to DONATED_KEY_PROVIDERS in task #12; later gated behind the
        donated-key toggle when the donated Gemini path got flaky.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "true")

        result = get_openrouter_api_key("openrouter/google/gemini-3.1-pro-preview")
        assert result == "special_key"

    def test_google_model_uses_general_key_when_donated_toggle_off(self, monkeypatch):
        """With the donated-key toggle off (default), Google calls use the general key."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "false")

        result = get_openrouter_api_key("openrouter/google/gemini-3.1-pro-preview")
        assert result == "general_key"

    def test_google_model_default_uses_general_key(self, monkeypatch):
        """Default (env var unset) matches the toggle-off behavior."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")
        monkeypatch.delenv("GEMINI_USE_DONATED_OPENROUTER_KEY", raising=False)

        result = get_openrouter_api_key("openrouter/google/gemini-3.1-pro-preview")
        assert result == "general_key"

    def test_non_donated_provider_uses_general_key(self, monkeypatch):
        """Providers NOT in DONATED_KEY_PROVIDERS (e.g. x-ai for Grok) use the general key."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("openrouter/x-ai/grok-4.1-fast")
        assert result == "general_key"

    def test_fallback_to_general_key_when_special_missing(self, monkeypatch):
        """Should fall back to general key when special key not available."""
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("openrouter/openai/gpt-5.1")
        assert result == "general_key"

    def test_non_openrouter_model_uses_general_key(self, monkeypatch):
        """Non-OpenRouter models should use general key."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("perplexity/sonar-reasoning-pro")
        assert result == "general_key"

    def test_returns_none_when_no_keys_available(self, monkeypatch):
        """Should return None when no API keys are available."""
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = get_openrouter_api_key("openrouter/openai/gpt-5.1")
        assert result is None
