"""Cria o baseline de voz de um paciente a partir de gravações em que ele fala normalmente.

    python scripts/voice_baseline.py data/real/normal_1.wav data/real/normal_2.wav data/real/normal_3.wav --patient P010

Recomendações:
  * 3 a 5 gravações de 15–30 s, falando de forma natural (ler um texto curto serve);
  * mesmo canal da análise: se a consulta vier pelo WhatsApp, o baseline também deve vir
    pelo WhatsApp — a compressão altera jitter e shimmer;
  * em produção, o baseline seria gravado na admissão do paciente.

Grava ``models/voice_baseline_<paciente>.json`` (fora do Git). O ``analyze_audio.py`` passa
a usá-lo automaticamente para esse paciente.
Aceita WAV/FLAC/OGG e, com ffmpeg instalado, também MP4/M4A/OPUS (áudio do WhatsApp).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hospital_monitor.audio.acoustic import FEATURE_NAMES, extract_features, load_audio  # noqa: E402
from hospital_monitor.audio.voice_analyzer import ReferenceProfile  # noqa: E402
from hospital_monitor.services import voice_baseline_path  # noqa: E402

def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("audios", type=Path, nargs="+")
    ap.add_argument("--patient", required=True)
    args = ap.parse_args()
    feats = []
    for p in args.audios:
        f = extract_features(load_audio(p))
        feats.append(f)
        print(f"{p.name}: {f.duration_s:.0f} s, jitter {f.jitter_pct:.2f}%, shimmer {f.shimmer_pct:.1f}%, "
              f"HNR {f.hnr_db:.1f} dB, {f.syllable_rate:.1f} sílabas/s")
    if len(feats) < 3:
        print("aviso: com menos de 3 gravações o baseline fica frágil; use os pisos de variação padrão.")
    ref = ReferenceProfile.from_features(feats, source=f"baseline real do paciente {args.patient}")
    out = voice_baseline_path(args.patient)
    ref.save(out)
    print(f"\nBaseline salvo em {out} (n={ref.n}, features: {len(FEATURE_NAMES)})")


if __name__ == "__main__":
    main()
