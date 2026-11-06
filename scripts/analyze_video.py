"""Analisa um vídeo REAL de fisioterapia ou de cirurgia.

Fisioterapia (pose → desvios → relatório):
    python scripts/analyze_video.py fisio caminho/video.mp4 --patient P010 --backend yolo      # 2D, pesos do GitHub
    python scripts/analyze_video.py fisio caminho/video.mp4 --patient P010 --backend mediapipe # 3D
    python scripts/analyze_video.py fisio keypoints.json --patient P010 --video caminho/video.mp4

    Câmera: corpo inteiro no quadro (cabeça aos pés), celular parado na altura do quadril.
    Com YOLOv8 (2D), prefira câmera de lado ou em diagonal leve; a assimetria é medida
    em relação à referência, então o mesmo posicionamento deve ser usado nos dois vídeos.

    Para criar a referência a partir de um vídeo de execução correta (grava em
    models/squat_reference_real.json, sem tocar na referência da demo):
    python scripts/analyze_video.py referencia caminho/execucao_correta.mp4

Cirurgia (YOLOv8 → regras → relatório):
    python scripts/analyze_video.py cirurgia caminho/video.mp4 --patient P020 --weights models/surgical_yolov8n.pt

Requer os extras de visão:  pip install -r requirements-ml.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from hospital_monitor import services  # noqa: E402
from hospital_monitor.video.exercise_analyzer import ExerciseAnalyzer, ExerciseReference, build_reference  # noqa: E402
from hospital_monitor.video.pose import get_pose_estimator, load_keypoints  # noqa: E402
from hospital_monitor.video.render import render_overlay, render_session  # noqa: E402
from hospital_monitor.video.report import exercise_report, surgical_report  # noqa: E402
from hospital_monitor.video.surgical import SurgicalVideoAnalyzer, YoloDetector  # noqa: E402


REAL_REF = ROOT / "models" / "squat_reference_real.json"


def _keypoints(path: Path, backend: str, weights: str | None = None):
    if path.suffix in (".json", ".npy"):
        return load_keypoints(path)
    kp, fps = get_pose_estimator(backend, weights).estimate_video(path)
    cache = path.with_suffix(".keypoints.json")  # reaproveitável sem rodar o modelo de novo
    cache.write_text(json.dumps({"fps": fps, "keypoints": np.where(np.isnan(kp), None, np.round(kp, 4)).tolist()}))
    return kp, fps


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
    ap.add_argument("--weights", default=None,
                    help="pesos YOLOv8: pose (padrão yolov8m-pose.pt) ou cirurgia (padrão models/surgical_yolov8n.pt)")
    ap.add_argument("--video", type=Path, default=None, help="vídeo original, para desenhar o esqueleto por cima")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "real")
    ap.add_argument("--reference", type=Path, default=REAL_REF,
                    help="referência do exercício (padrão: a gravada pelo modo 'referencia')")
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "referencia":
        kp, fps = _keypoints(args.path, args.backend, args.weights)
        ref = build_reference(kp, fps)
        ref.save(REAL_REF)
        print(f"referência salva em {REAL_REF.name}: {ref.n_reps} repetições, ROM do joelho {ref.rom:.0f}°")
    elif args.mode == "fisio":
        kp, fps = _keypoints(args.path, args.backend, args.weights)
        if args.reference.exists():
            analyzer = ExerciseAnalyzer(ExerciseReference.load(args.reference))
        else:
            print("aviso: sem referência gravada; usando a referência sintética da demo. "
                  "Rode antes o modo 'referencia' com um vídeo de execução correta.")
            analyzer = services.make_exercise_analyzer()
        res = analyzer.analyze(kp, fps, args.patient, now)
        print(f"{len(res.reps)} repetições encontradas")
        rep = exercise_report(res, kp, args.out / f"fisio_{args.patient}.html", args.path.name, now)
        source = args.video or (args.path if args.path.suffix not in (".json", ".npy") else None)
        if source:
            render_overlay(source, kp, fps, args.out / f"fisio_{args.patient}.mp4", res.deviations)
        else:
            render_session(kp, fps, args.out / f"fisio_{args.patient}.mp4", res.deviations)
        for e in res.events:
            print(e.severity.value, e.description)
        print("relatório:", rep)
    else:
        dets = YoloDetector(args.weights or "models/surgical_yolov8n.pt").detect_video(args.path, fps=5)
        res = SurgicalVideoAnalyzer().analyze(dets, 5.0, args.patient, now)
        rep = surgical_report(res, args.out / f"cirurgia_{args.patient}.html", args.path.name, now)
        for f in res.findings:
            print(f.kind, f.description)
        print("relatório:", rep)


if __name__ == "__main__":
    main()
