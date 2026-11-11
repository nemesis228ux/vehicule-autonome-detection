"""Compatibility launcher for the detection demo."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .detection_app.cli import main, parse_args
    from .detection_app.rendering import RunStats
    from .detection_app.runtime import run
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.detection_app.cli import main, parse_args
    from src.detection_app.rendering import RunStats
    from src.detection_app.runtime import run


if __name__ == "__main__":
    raise SystemExit(main())
