"""Roda só a avaliação quantitativa e grava outputs/metrics.json.

    python scripts/evaluate.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hospital_monitor.evaluation import evaluate_all  # noqa: E402

if __name__ == "__main__":
    m = evaluate_all()
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(m, ensure_ascii=False, indent=1, default=float))
