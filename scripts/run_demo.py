"""Demo ponta a ponta do monitoramento multimodal.

Simula um dia de internação (28/10/2026, horário de Brasília) com 4 pacientes:

  P001  Leito 12, pós-operatório — agitação noturna, erro de prescrição,
        fisioterapia com assimetria e dessaturação + consulta com dispneia.
  P002  Leito 07, ortopedia — queda durante a fisioterapia; consulta normal.
  P003  Leito 15, neurologia — taquicardia, duplicidade de anticoagulante,
        consulta com disartria (fala + relato) e imobilidade à tarde.
  P004  Centro cirúrgico — vídeo do procedimento com gaze não contabilizada.

Executa todos os pipelines, faz o replay dos eventos em ordem temporal pelo
barramento → fusão → alertas, gera relatórios e as métricas de avaliação.

Uso:
    python scripts/run_demo.py                 # tudo, saída em outputs/
    python scripts/run_demo.py --realtime 0.8  # replay com pausa (para gravar o vídeo)
    python scripts/run_demo.py --skip-eval     # pula a avaliação quantitativa
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from hospital_monitor import services  # noqa: E402
from hospital_monitor.alerts.notifiers import ConsoleNotifier, FileNotifier, notifiers_from_env  # noqa: E402
from hospital_monitor.anomaly.movement import MovementAnalyzer  # noqa: E402
from hospital_monitor.config import load_settings  # noqa: E402
from hospital_monitor.data import synthetic as syn  # noqa: E402
from hospital_monitor.orchestrator import Orchestrator  # noqa: E402
from hospital_monitor.video.render import render_session  # noqa: E402
from hospital_monitor.video.report import exercise_report, surgical_report  # noqa: E402
from hospital_monitor.video.surgical import SurgicalVideoAnalyzer  # noqa: E402

TZ = ZoneInfo("America/Sao_Paulo")
DAY = datetime(2026, 10, 28, tzinfo=TZ)
log = logging.getLogger("demo")

PATIENTS = {
    "P001": "Leito 12 · pós-operatório",
    "P002": "Leito 07 · ortopedia",
    "P003": "Leito 15 · neurologia",
    "P004": "Centro cirúrgico · sala 3",
}


def at(h: int, m: int = 0, day: datetime = DAY) -> datetime:
    return day + timedelta(hours=h, minutes=m)


def run_vitals(settings, out: Path):
    det = services.make_vitals_detector(settings)
    scenario = {
        "P001": [syn.VitalAnomaly("desaturation", 14 * 60, 60)],
        "P002": [],
        "P003": [syn.VitalAnomaly("tachycardia", 9 * 60 + 40, 40)],
    }
    frames, events = {}, []
    for i, (pid, an) in enumerate(scenario.items()):
        df = syn.generate_vitals(pid, start=DAY, hours=24, anomalies=an, seed=10 + i)
        scored, evs = det.process_frame(df)
        frames[pid] = scored
        events += evs
        scored.to_csv(out / "data" / f"vitais_{pid}.csv", index=False)
    return frames, events


def run_prescriptions(out: Path):
    an = services.make_prescription_analyzer()
    start = DAY - timedelta(days=2) + timedelta(hours=8)
    frames, events, findings = {}, [], []
    for pid, anomalies, seed in [("P001", [{"kind": "decimal_error", "at_hours": 52}], 1),
                                 ("P002", [], 2),
                                 ("P003", [{"kind": "duplication", "at_hours": 51.25}], 3)]:
        df = syn.generate_prescriptions(pid, start=start, days=3, anomalies=anomalies, seed=seed)
        f, evs, fi = an.analyze(df)
        frames[pid] = f
        events += evs
        findings += [(pid, x) for x in fi]
        df.to_csv(out / "data" / f"prescricoes_{pid}.csv", index=False)
    return frames, events, findings


def run_movement(out: Path):
    an = MovementAnalyzer()
    start = DAY - timedelta(days=1)
    frames, events = {}, []
    for pid, anomalies, seed in [("P001", [{"kind": "night_agitation", "start_h": 26, "duration_h": 1.5}], 4),
                                 ("P002", [], 5),
                                 ("P003", [{"kind": "immobility", "start_h": 38, "duration_h": 3}], 6)]:
        df = syn.generate_movement(pid, start=start, hours=48, anomalies=anomalies, seed=seed)
        f, evs = an.analyze(df)
        frames[pid] = f
        events += evs
        df.to_csv(out / "data" / f"movimentacao_{pid}.csv", index=False)
    return frames, events


def run_exercise(out: Path):
    an = services.make_exercise_analyzer()
    results, events = {}, []
    for pid, kwargs, t0, seed in [
        ("P001", dict(reps=6, deviation="asymmetric", deviation_reps=(1, 3)), at(10, 0), 21),
        ("P002", dict(reps=6, fall_at_rep=3), at(9, 0), 22),
    ]:
        s = syn.generate_squat_session(seed=seed, **kwargs)
        res = an.analyze(s.keypoints, s.fps, pid, t0)
        kp_path = out / "data" / f"keypoints_fisio_{pid}.json"
        kp_path.write_text(json.dumps({"fps": s.fps, "keypoints": np.round(s.keypoints, 4).tolist(),
                                       "labels": s.labels}), encoding="utf-8")
        exercise_report(res, s.keypoints, out / "video" / f"relatorio_fisioterapia_{pid}.html",
                        "Agachamento assistido — 6 repetições", t0)
        render_session(s.keypoints, s.fps, out / "video" / f"fisioterapia_{pid}.mp4", res.deviations)
        results[pid] = res
        events += res.events
    return results, events


def run_surgery(out: Path):
    dets, _ = syn.generate_surgical_detections(minutes=6, fps=5, scenario="retained_gauze", seed=31)
    t0 = at(7, 30)
    res = SurgicalVideoAnalyzer().analyze(dets, 5.0, "P004", t0)
    (out / "data" / "deteccoes_yolo_cirurgia_P004.json").write_text(json.dumps({"fps": 5, "frames": dets}),
                                                                    encoding="utf-8")
    surgical_report(res, out / "video" / "relatorio_cirurgia_P004.html", "Laparotomia exploradora (simulada)", t0)
    return res, res.events


def run_audio(out: Path):
    pipe = services.make_audio_pipeline("auto")
    results, events = {}, []
    for pid, cond, t, seed in [("P003", "dysarthria", at(10, 5), 41), ("P002", "normal", at(10, 30), 42),
                               ("P001", "dyspnea", at(14, 20), 43)]:
        y = syn.synthesize_voice(cond, seconds=14, seed=seed)
        wav = out / "data" / "audio" / f"consulta_{pid}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav, y, 16000, subtype="PCM_16")
        wav.with_suffix(".txt").write_text(syn.CONSULT_TRANSCRIPTS[cond], encoding="utf-8")
        res = pipe.analyze(wav, pid, t, y=y)
        rel = str(wav.relative_to(out))  # caminho relativo: saída portátil
        res.audio_path = rel
        for e in res.events:
            if "audio" in e.evidence:
                e.evidence["audio"] = rel
        (out / "audio").mkdir(exist_ok=True)
        (out / "audio" / f"analise_{pid}.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1,
                                                                      default=str), encoding="utf-8")
        results[pid] = res
        events += res.events
    return results, events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--realtime", type=float, default=0.0, help="segundos de pausa entre eventos no replay")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="não imprime alertas no terminal")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for d in ("data", "video", "audio", "figuras"):
        (out / d).mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    log.info("1/6 Modelos e referências…")
    services.ensure_models()

    log.info("2/6 Pipelines: sinais vitais, prescrições, movimentação…")
    vit_frames, ev_vit = run_vitals(settings, out)
    rx_frames, ev_rx, rx_findings = run_prescriptions(out)
    mv_frames, ev_mv = run_movement(out)
    log.info("3/6 Pipeline de vídeo: fisioterapia (pose + DTW) e cirurgia (YOLOv8 + regras)…")
    ex_results, ev_ex = run_exercise(out)
    sx_result, ev_sx = run_surgery(out)
    log.info("4/6 Pipeline de áudio: transcrição, voz e NLP clínico…")
    au_results, ev_au = run_audio(out)

    all_events = ev_vit + ev_rx + ev_mv + ev_ex + ev_sx + ev_au
    log.info("5/6 Replay de %d eventos → fusão multimodal → alertas", len(all_events))
    notifiers = [FileNotifier(out / "alerts.jsonl"), *notifiers_from_env()]
    if not args.quiet:
        notifiers.insert(0, ConsoleNotifier())
    orch = Orchestrator(settings, notifiers, out)
    orch.replay(all_events, pause_s=args.realtime)
    # reconhecimento do alerta de dessaturação pela equipe (fecha o ciclo)
    for a in orch.alerts.list():
        if a.patient_id == "P001" and a.level == "critical" and "sinais vitais" in a.title + " ".join(a.details).lower():
            orch.alerts.acknowledge(a.alert_id, "Enf. plantão (posto 2)", a.created_at + timedelta(minutes=4))
            break
    orch.dump()

    metrics = {}
    if not args.skip_eval:
        log.info("6/6 Avaliação quantitativa dos detectores (dados sintéticos rotulados)…")
        from hospital_monitor.evaluation import evaluate_all

        metrics = evaluate_all()
        mfile = ROOT / "models" / "voice_rf_metrics.json"
        if mfile.exists():
            metrics["voz_classificador"] = json.loads(mfile.read_text(encoding="utf-8"))
        (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    from report_builder import build_demo_report  # noqa: E402

    path = build_demo_report(out, orch, vit_frames, rx_frames, rx_findings, mv_frames, ex_results, sx_result,
                             au_results, metrics, PATIENTS)
    alerts = orch.alerts.list()
    log.info("\nResumo: %d eventos, %d alertas (%d críticos, %d atenção), %d atualizações, %d duplicados suprimidos.",
             len(all_events), len(alerts), sum(a.level == "critical" for a in alerts),
             sum(a.level == "attention" for a in alerts), orch.alerts.updates, orch.alerts.suppressed)
    log.info("Relatório: %s", path)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
