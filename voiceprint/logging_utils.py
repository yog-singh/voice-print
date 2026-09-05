import logging
import os

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=os.environ.get("VOICEPRINT_LOG", "WARNING").upper(),
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def set_level(level: str) -> None:
    get_logger("voiceprint")
    logging.getLogger("voiceprint").setLevel(level.upper())
