import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


def set_up_logging(log_level: int) -> None:
    custom_theme = Theme({
        "logging.level.debug": "cyan",
        "logging.level.info": "blue",
        "logging.level.warning": "magenta",
        "logging.level.error": "red bold",
        "logging.level.critical": "white on red",
    })
    console = Console(theme=custom_theme, force_terminal=True)

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_path=True,
            markup=False,
            enable_link_path=False
        )]
    )
    logging.debug(f"Logging configured with log level: {log_level}")