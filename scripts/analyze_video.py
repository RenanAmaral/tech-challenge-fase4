"""Analisa um vídeo REAL de fisioterapia ou de cirurgia.

Fisioterapia (pose → desvios → relatório):
    python scripts/analyze_video.py fisio caminho/video.mp4 --patient P010 --backend mediapipe
    python scripts/analyze_video.py fisio keypoints.json --patient P010        # keypoints já extraídos

    Para criar a referência a partir de um vídeo de execução correta (grava em
    models/squat_reference_real.json, sem tocar na referência da demo):
    python scripts/analyze_video.py referencia caminho/execucao_correta.mp4

Cirurgia (YOLOv8 → regras → relatório):
    python scripts/analyze_video.py cirurgia caminho/video.mp4 --patient P020 --weights models/surgical_yolov8n.pt

Requer os extras de visão:  pip install -r requirements-ml.txt
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hospital_monitor import services  # noqa: E402
from hospital_monitor.video.exercise_analyzer import ExerciseAnalyzer, ExerciseReference, build_reference  # noqa: E402
from hospital_monitor.video.pose import get_pose_estimator, load_keypoints  # noqa: E402
from hospital_monitor.video.render import render_session  # noqa: E402
from hospital_monitor.video.report import exercise_report, surgical_report  # noqa: E402
from hospital_monitor.video.surgical import SurgicalVideoAnalyzer, YoloDetector  # noqa: E402


REAL_REF = ROOT / "models" / "squat_reference_real.json"


def _keypoints(path: Path, backend: str):
    if path.suffix in (".json", ".npy"):
        return load_keypoints(path)
    return get_pose_estimator(backend).estimate_video(path)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # acentos no console do Windows
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fisio", "referencia", "cirurgia"])
    ap.add_argument("path", type=Path)
    ap.add_argument("--patient", default="P000")
    ap.add_argument("--backend", choices=["mediapipe", "yolo"], default="mediapipe")
    ap.add_argument("--weights", default="models/surgical_yolov8n.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "real")
    ap.add_argument("--reference", type=Path, default=REAL_REF,
                    help="referência do exercício (padrão: a gravada pelo modo 'referencia')")
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "referencia":
        kp, fps = _keypoints(args.path, args.backend)
        ref = build_reference(kp, fps)
        ref.save(REAL_REF)
        print(f"referência salva em {REAL_REF.name}: {ref.n_reps} repetições, ROM do joelho {ref.rom:.0f}°")
    elif args.mode == "fisio":
        kp, fps = _keypoints(args.path, args.backend)
        if args.reference.exists():
            analyzer = ExerciseAnalyzer(ExerciseReference.load(args.reference))
        else:
            print("aviso: sem referência gravada; usando a referência sintética da demo. "
                  "Rode antes o modo 'referencia' com um vídeo de execução correta.")
            analyzer = services.make_exercise_analyzer()
        res = analyzer.analyze(kp, fps, args.patient, now)
        print(f"{len(res.reps)} repetições encontradas")
        rep = exercise_report(res, kp, args.out / f"fisio_{args.patient}.html", args.path.name, now)
        render_session(kp, fps, args.out / f"fisio_{args.patient}.mp4", res.deviations)
        for e in res.events:
            print(e.severity.value, e.description)
        print("relatório:", rep)
    else:
        dets = YoloDetector(args.weights).detect_video(args.path, fps=5)
        res = SurgicalVideoAnalyzer().analyze(dets, 5.0, args.patient, now)
        rep = surgical_report(res, args.out / f"cirurgia_{args.patient}.html", args.path.name, now)
        for f in res.findings:
            print(f.kind, f.description)
        print("relatório:", rep)


if __name__ == "__main__":
    main()
