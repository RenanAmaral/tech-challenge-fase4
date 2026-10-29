"""Analisa uma gravação REAL de consulta (WAV/FLAC/OGG).

    python scripts/analyze_audio.py consulta.wav --patient P010
    python scripts/analyze_audio.py consulta.wav --patient P010 --transcriber sidecar   # usa consulta.txt

Com ``faster-whisper`` instalado (requirements-ml.txt) a transcrição é feita
localmente; sem ele, coloque a transcrição em ``consulta.txt`` ao lado do áudio.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hospital_monitor import services  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--patient", default="P000")
    ap.add_argument("--transcriber", choices=["auto", "whisper", "sidecar"], default="auto")
    args = ap.parse_args()
    res = services.make_audio_pipeline(args.transcriber).analyze(args.audio, args.patient, datetime.now(timezone.utc))
    d = res.to_dict()
    print(json.dumps({k: d[k] for k in ("transcript", "voice_assessment", "text_analysis", "events")},
                     ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
