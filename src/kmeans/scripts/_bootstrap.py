"""Put the repo ``src/`` dir on sys.path so ``import kmeans`` / ``import sragents``
work whether or not the package is pip-installed. Imported for its side effect.
"""

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]   # .../src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
