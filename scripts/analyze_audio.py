"""Analisa uma gravação REAL de consulta (WAV/FLAC/OGG).

    python scripts/analyze_audio.py consulta.wav --patient P010
    python scripts/analyze_audio.py consulta.wav --patient P010 --transcriber sidecar   # usa consulta.txt
    python scripts/analyze_audio.py consulta.wav --patient P010 --full                  # JSON completo no console

Com ``faster-whisper`` instalado (requirements-ml.txt) a transcrição é feita
localmente; sem ele, coloque a transcrição em ``consulta.txt`` (UTF-8) ao lado
do áudio. O JSON completo sempre é gravado em ``outputs/real/audio_<paciente>.json``
e o console mostra um resumo legível.
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
from hospital_monitor.reporting import kind_pt  # noqa: E402


def _utf8_console() -> None:
    """O console do Windows usa cp1252 por padrão e quebra os acentos."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _summary(d: dict) -> str:
    v, t = d["voice_assessment"], d["text_analysis"]
    lines = [
        f"Transcrição ({d['transcript']['backend']}): {d['transcript']['text']}",
        "",
        f"Voz: {v['main_condition'] or 'sem alteração'} (escore {v['anomaly_score']:.2f})",
        "  escores: " + ", ".join(f"{k} {s:.2f}" for k, s in v["scores"].items()),
    ]
    if v["drivers"]:
        lines.append("  maiores desvios: " + "; ".join(v["drivers"]))
    if v.get("classifier_probs"):
        lines.append("  Random Forest: " + ", ".join(f"{k} {p:.2f}" for k, p in v["classifier_probs"].items()))
    affirmed = [f["term"] for f in t["findings"] if f["effective_severity"] > 0]
    negated = [f["term"] for f in t["findings"] if f["negated"]]
    lines += [
        "",
        f"Texto: escore {t['anomaly_score']:.2f} · sentimento {t['sentiment']['label']} "
        f"({t['sentiment']['compound']:+.2f}, {t['sentiment']['backend']})",
        "  termos afirmados: " + (", ".join(dict.fromkeys(affirmed)) or "—"),
        "  termos negados: " + (", ".join(dict.fromkeys(negated)) or "—"),
        "  sinais de alarme: " + (", ".join(t["critical_terms"]) or "—"),
        "",
        f"Eventos emitidos: {len(d['events'])}",
    ]
    for e in d["events"]:
        lines.append(f"  [{e['severity']}] {kind_pt(e['kind'])} ({e['anomaly_score']:.2f}): {e['description']}")
    return "\n".join(lines)


def main() -> None:
    _utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--patient", default="P000")
    ap.add_argument("--transcriber", choices=["auto", "whisper", "sidecar"], default="auto")
    ap.add_argument("--full", action="store_true", help="imprime o JSON completo em vez do resumo")
    args = ap.parse_args()
    res = services.make_audio_pipeline(args.transcriber).analyze(args.audio, args.patient, datetime.now(timezone.utc))
    d = res.to_dict()
    out = ROOT / "outputs" / "real" / f"audio_{args.patient}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if args.full:
        print(json.dumps({k: d[k] for k in ("transcript", "voice_assessment", "text_analysis", "events")},
                         ensure_ascii=False, indent=1, default=str))
    else:
        print(_summary(d))
    print(f"\nJSON completo: {out}")


if __name__ == "__main__":
    main()
