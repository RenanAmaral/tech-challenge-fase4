"""(Re)gera os artefatos de modelo em models/ e imprime as métricas de treino.

    python scripts/train_models.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hospital_monitor.services import ensure_models  # noqa: E402

if __name__ == "__main__":
    info = ensure_models(force=True)
    print(json.dumps(info, ensure_ascii=False, indent=1, default=str))
    print("artefatos em", ROOT / "models")
