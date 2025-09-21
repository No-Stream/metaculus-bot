import logging

from metaculus_bot import config


def _reset_env_loaded(monkeypatch) -> None:
    monkeypatch.setattr(config, "_ENV_LOADED", False, raising=False)


def test_load_environment_idempotent(monkeypatch):
    _reset_env_loaded(monkeypatch)

    calls: list[tuple[str | None, bool]] = []

    def fake_load(path: str | None = None, override: bool = False) -> bool:  # noqa: D401 - simple stub
        calls.append((path, override))
        return True

    monkeypatch.setattr(config, "load_dotenv", fake_load)

    config.load_environment()
    config.load_environment()

    assert calls == [(None, False), (".env.local", True)]


def test_load_environment_handles_failure_once(monkeypatch, caplog):
    _reset_env_loaded(monkeypatch)

    caplog.set_level(logging.WARNING)
    call_count = {"value": 0}

    def failing_load(*_args, **_kwargs):  # noqa: D401 - simple stub
        call_count["value"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(config, "load_dotenv", failing_load)

    config.load_environment()
    config.load_environment()

    assert call_count["value"] == 1
    assert "Failed to load environment files" in caplog.text
