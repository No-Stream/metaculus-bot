from metaculus_bot.forecaster import TemplateForecaster  # noqa: F401  # re-export for back-compat

if __name__ == "__main__":
    from metaculus_bot.cli import main as cli_main

    cli_main()
