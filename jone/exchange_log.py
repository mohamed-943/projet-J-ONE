"""Journal des échanges (EF-12) : une ligne JSON par énoncé traité, dans
logs/echanges-AAAAMMJJ.jsonl. C'est la source du rapport de tests
(tools/rapport_tests.py).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import threading
from datetime import datetime
from pathlib import Path


class ExchangeLogger:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        now = datetime.now().astimezone()
        record = {"horodatage": now.isoformat(timespec="milliseconds"), **record}
        path = self.directory / f"echanges-{now:%Y%m%d}.jsonl"
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def setup_logging(directory: Path, level: str = "INFO") -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file_handler = logging.handlers.RotatingFileHandler(
        directory / "jone.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(file_handler)
    for noisy in ("httpx", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
