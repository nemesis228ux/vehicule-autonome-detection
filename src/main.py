"""Point d'entrée principal de l'application de détection."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

if __package__:
    from .demo_detection import main as run_detection
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.demo_detection import main as run_detection


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Lance le pipeline de détection avec les options de la ligne de commande."""
    return run_detection(argv)


if __name__ == "__main__":
    raise SystemExit(main())
