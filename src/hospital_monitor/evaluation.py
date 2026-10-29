"""Avaliação quantitativa de cada detector com dados sintéticos rotulados.

Gera as tabelas de resultados do relatório técnico. Todos os experimentos são
determinísticos (seeds fixos) e reproduzíveis com ``python scripts/evaluate.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .anomaly.metrics import event_level, pointwise
from .anomaly.movement import MovementAnalyzer
from .anomaly.vitals import IsolationForestVitals, StreamingVitalsDetector, _confirm
from .audio.clinical_nlp import extract_findings
from .data import synthetic as syn
from .services import make_exercise_analyzer, make_prescription_analyzer
from .video.surgical import SurgicalVideoAnalyzer

T0 = datetime(2026, 10, 28, tzinfo=timezone.utc)
VITAL_SCENARIO = [
    syn.VitalAnomaly("tachycardia", 5 * 60, 40),
    syn.VitalAnomaly("hypotension", 9 * 60, 45),
    syn.VitalAnomaly("desaturation", 14 * 60, 60),
    syn.VitalAnomaly("silent_deterioration", 18 * 60 + 30, 150),
    syn.VitalAnomaly("fever", 22 * 60, 90),
]


def evaluate_vitals(seeds=range(10)) -> dict:
    rows = {k: [] for k in ("limiares", "z_baseline", "multivariado", "combinado", "isolation_forest")}
    per_type: dict[str, list[int]] = {}
    fp_normal = 0
    for sd in seeds:
        df = syn.generate_vitals(f"E{sd}", anomalies=VITAL_SCENARIO, seed=sd)
        out, _ = StreamingVitalsDetector().process_frame(df)
        preds = {
            "limiares": _confirm(out["threshold_score"].to_numpy() >= 0.5, 2),
            "z_baseline": _confirm(out["z_score"].to_numpy() >= 0.5, 2),
            "multivariado": _confirm(out["multi_score"].to_numpy() >= 0.5, 2),
            "combinado": out["flag"].to_numpy(),
            "isolation_forest": _confirm(IsolationForestVitals(seed=sd).fit(df.iloc[:240]).predict(df), 2),
        }
        for k, p in preds.items():
            rows[k].append((pointwise(out["label"], p), event_level(out["label"], p)))
        for kind in df["anomaly_type"].unique():
            if not kind:
                continue
            m = (df["anomaly_type"] == kind).to_numpy()
            per_type.setdefault(kind, []).append(int((preds["combinado"] & m).any()))
        # série sem anomalias: quantos episódios falsos?
        normal, _ = StreamingVitalsDetector().process_frame(syn.generate_vitals(f"N{sd}", seed=100 + sd))
        fp_normal += event_level(normal["label"], normal["flag"])["fp"]
    table = []
    for k, vals in rows.items():
        table.append({
            "detector": k,
            "precision_pontual": round(np.mean([v[0].precision for v in vals]), 3),
            "recall_pontual": round(np.mean([v[0].recall for v in vals]), 3),
            "f1_pontual": round(np.mean([v[0].f1 for v in vals]), 3),
            "precision_episodio": round(np.mean([v[1]["precision"] for v in vals]), 3),
            "recall_episodio": round(np.mean([v[1]["recall"] for v in vals]), 3),
            "atraso_medio_min": round(np.mean([v[1]["mean_delay_min"] or 0 for v in vals]), 1),
            "falsos_episodios_por_dia": round(np.mean([v[1]["fp"] for v in vals]), 2),
        })
    return {
        "n_series": len(list(seeds)),
        "detectores": table,
        "recall_por_tipo": {k: round(float(np.mean(v)), 2) for k, v in per_type.items()},
        "falsos_episodios_em_series_normais_por_dia": round(fp_normal / len(list(seeds)), 2),
    }


def evaluate_prescriptions(n: int = 10) -> dict:
    an = make_prescription_analyzer()
    kinds = ["decimal_error", "above_max", "duplication", "abrupt_change"]
    frames = []
    for sd in range(n):
        anomalies = [{"kind": k, "at_hours": 26 + 9 * i} for i, k in enumerate(kinds)]
        f, _, _ = an.analyze(syn.generate_prescriptions(f"R{sd}", anomalies=anomalies, seed=500 + sd))
        frames.append(f)
    f = pd.concat(frames)
    rules = pointwise(f["label"], f["rule_score"] >= 0.5).to_dict()
    iforest = pointwise(f["label"], f["if_flag"]).to_dict()
    by_kind = {k: round(float((f[f["anomaly_type"] == k]["flag"]).mean()), 2) for k in kinds}
    return {"n_prescricoes": len(f), "n_anomalas": int(f["label"].sum()), "regras": rules,
            "isolation_forest_isolado": iforest, "recall_por_tipo": by_kind}


def evaluate_movement(n: int = 10) -> dict:
    res, fp_normal = [], 0
    for sd in range(n):
        an = [{"kind": "immobility", "start_h": 10 + sd % 3, "duration_h": 2.5 + 0.2 * sd},
              {"kind": "night_agitation", "start_h": 25 + sd % 3, "duration_h": 1.0 + 0.1 * sd},
              {"kind": "fall", "start_h": 38 + sd % 5}]
        out, _ = MovementAnalyzer().analyze(syn.generate_movement(f"M{sd}", anomalies=an, seed=sd))
        res.append(event_level(out["label"], out["pred"], step_minutes=5))
        normal, ev = MovementAnalyzer().analyze(syn.generate_movement(f"MN{sd}", seed=900 + sd))
        fp_normal += len(ev)
    return {"n_series": n, "precision_episodio": round(np.mean([r["precision"] for r in res]), 3),
            "recall_episodio": round(np.mean([r["recall"] for r in res]), 3),
            "atraso_medio_min": round(np.mean([r["mean_delay_min"] or 0 for r in res]), 1),
            "falsos_alarmes_em_48h_normais": fp_normal}


def evaluate_exercise(n: int = 8) -> dict:
    an = make_exercise_analyzer()
    out = {}
    fp_reps = total_normal_reps = 0
    for dev in ("shallow", "asymmetric", "valgus", "trunk_lean"):
        tp = fn = 0
        for sd in range(n):
            s = syn.generate_squat_session(reps=6, deviation=dev, deviation_reps=(1, 4), seed=300 + sd)
            r = an.analyze(s.keypoints, s.fps, "X", T0)
            flagged = {(d.kind, d.rep) for d in r.deviations}
            for rep in (1, 4):
                tp += (dev, rep) in flagged
                fn += (dev, rep) not in flagged
            fp_reps += sum(1 for d in r.deviations if d.rep not in (1, 4))
            total_normal_reps += 4
        out[dev] = round(tp / (tp + fn), 3)
    def has_fall(session) -> bool:
        return any(d.kind == "fall" for d in an.analyze(session.keypoints, session.fps, "X", T0).deviations)

    falls = sum(has_fall(syn.generate_squat_session(reps=5, fall_at_rep=2, seed=700 + sd)) for sd in range(n))
    false_falls = sum(has_fall(syn.generate_squat_session(reps=5, seed=800 + sd)) for sd in range(n))
    return {"recall_por_desvio": out, "reps_normais_marcadas": f"{fp_reps}/{total_normal_reps}",
            "quedas_detectadas": f"{falls}/{n}", "quedas_falsas_em_sessoes_normais": f"{false_falls}/{n}"}


def evaluate_surgical(n: int = 5) -> dict:
    an = SurgicalVideoAnalyzer()
    hit = {"instrument_outside_field": 0, "camera_occlusion": 0, "gauze_count_mismatch": 0}
    fp = 0
    for sd in range(n):
        dets, labels = syn.generate_surgical_detections(scenario="retained_gauze", seed=sd)
        kinds = {f.kind for f in an.analyze(dets, 5.0, "S", T0).findings}
        for k in hit:
            hit[k] += k in kinds
        dets, _ = syn.generate_surgical_detections(scenario="normal", seed=50 + sd)
        fp += len(an.analyze(dets, 5.0, "S", T0).findings)
    return {"n_videos_com_falha": n, "detectados": {k: f"{v}/{n}" for k, v in hit.items()},
            "achados_em_videos_normais": f"{fp} em {n}"}


NLP_CASES = [
    ("Estou com falta de ar desde ontem.", "dispneia", False),
    ("Não tenho falta de ar.", "dispneia", True),
    ("Nega dor no peito.", "dor torácica", True),
    ("Sinto uma pressão no peito quando subo escadas.", "dor torácica", False),
    ("Sem febre há dois dias.", "febre", True),
    ("Tive febre à noite.", "febre", False),
    ("Minha fala está enrolada.", "disartria", False),
    ("Ele não está com a fala enrolada, mas sente tontura.", "disartria", True),
    ("Ele não está com a fala enrolada, mas sente tontura.", "tontura", False),
    ("Desmaiei no banheiro hoje cedo.", "síncope", False),
    ("Nunca desmaiei.", "síncope", True),
    ("Estou muito cansado.", "fadiga", False),
    ("Não sinto cansaço nem tontura.", "fadiga", True),
    ("Não sinto cansaço nem tontura.", "tontura", True),
    ("Talvez seja febre, não sei.", "febre", False),
    ("Sinto fraqueza no braço direito.", "déficit motor", False),
    ("O rosto está meio dormente.", "parestesia facial", False),
    ("Não tenho dor de cabeça, porém estou com enjoo.", "cefaleia", True),
    ("Não tenho dor de cabeça, porém estou com enjoo.", "náusea", False),
    ("Estou sangrando pelo curativo.", "hemorragia", False),
]


def evaluate_nlp() -> dict:
    found = neg_ok = 0
    errors = []
    for text, term, negated in NLP_CASES:
        f = [x for x in extract_findings(text) if x.term == term]
        if f:
            found += 1
            if f[0].negated == negated:
                neg_ok += 1
            else:
                errors.append(text)
        else:
            errors.append(text)
    n = len(NLP_CASES)
    return {"n_frases": n, "termos_encontrados": f"{found}/{n}", "negacao_correta": f"{neg_ok}/{n}",
            "acuracia": round(neg_ok / n, 3), "erros": errors}


def _py(x):
    """Converte tipos NumPy em tipos nativos (JSON limpo)."""
    if isinstance(x, dict):
        return {k: _py(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_py(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def evaluate_all() -> dict:
    return _py({"sinais_vitais": evaluate_vitals(), "prescricoes": evaluate_prescriptions(),
                "movimentacao": evaluate_movement(), "video_fisioterapia": evaluate_exercise(),
                "video_cirurgia": evaluate_surgical(), "nlp_clinico": evaluate_nlp()})
