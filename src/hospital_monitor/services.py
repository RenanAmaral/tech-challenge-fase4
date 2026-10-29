"""Montagem dos pipelines com seus artefatos de modelo.

Artefatos em ``models/`` (gerados por ``scripts/train_models.py``):
  * ``voice_reference.json`` — perfil acústico de falantes saudáveis;
  * ``voice_rf.joblib`` — Random Forest para diagnóstico diferencial da voz;
  * ``squat_reference.json`` — padrão do agachamento correto (DTW);
  * ``prescription_iforest.joblib`` — Isolation Forest + limiar.

Se um artefato não existir, é construído na hora a partir de dados sintéticos.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .anomaly.prescriptions import PrescriptionAnalyzer, build_features
from .anomaly.vitals import StreamingVitalsDetector
from .audio.acoustic import FEATURE_NAMES, extract_features
from .audio.pipeline import AudioPipeline
from .audio.transcribe import get_transcriber
from .audio.voice_analyzer import ReferenceProfile, VoiceAnalyzer
from .config import ROOT, Settings
from .data import synthetic as syn
from .video.exercise_analyzer import ExerciseAnalyzer, ExerciseReference, build_reference

log = logging.getLogger(__name__)
MODELS = ROOT / "models"


def build_voice_reference(n: int = 12, seed: int = 1000) -> ReferenceProfile:
    feats = [extract_features(syn.synthesize_voice("normal", seed=seed + i, speaker_variation=True)) for i in range(n)]
    return ReferenceProfile.from_features(feats, source=f"synthetic-healthy (n={n})")


def train_voice_classifier(n_per_class: int = 30, seed: int = 2000):
    """Random Forest nas features acústicas; devolve (modelo, acurácia CV, matriz de confusão)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X, y = [], []
    for ci, cond in enumerate(["normal", "fatigue", "dysarthria", "dyspnea"]):
        for i in range(n_per_class):
            f = extract_features(syn.synthesize_voice(cond, seed=seed + 100 * ci + i, speaker_variation=True))
            X.append(f.vector(FEATURE_NAMES))
            y.append({"normal": "normal", "fatigue": "fadiga", "dysarthria": "disartria", "dyspnea": "dispneia"}[cond])
    X, y = np.array(X), np.array(y)
    clf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")
    pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    labels = ["normal", "fadiga", "disartria", "dispneia"]
    cm = confusion_matrix(y, pred, labels=labels)
    acc = float((pred == y).mean())
    clf.fit(X, y)
    importances = dict(sorted(((n, round(float(v), 3)) for n, v in zip(FEATURE_NAMES, clf.feature_importances_)),
                              key=lambda kv: -kv[1]))
    return clf, acc, {"labels": labels, "matrix": cm.tolist(), "importances": importances, "n": len(y)}


def build_squat_reference(seed: int = 100) -> ExerciseReference:
    s = syn.generate_squat_session(reps=8, seed=seed)
    return build_reference(s.keypoints, s.fps)


def train_prescription_iforest(n_patients: int = 40):
    hist = pd.concat([build_features(syn.generate_prescriptions(f"H{i:03d}", seed=i)) for i in range(n_patients)])
    return PrescriptionAnalyzer.fit_iforest(hist)


def ensure_models(force: bool = False) -> dict:
    """Garante os artefatos em ``models/``; devolve métricas de treino quando treina."""
    MODELS.mkdir(exist_ok=True)
    info: dict = {}
    if force or not (MODELS / "voice_reference.json").exists():
        build_voice_reference().save(MODELS / "voice_reference.json")
    if force or not (MODELS / "voice_rf.joblib").exists():
        clf, acc, cm = train_voice_classifier()
        joblib.dump(clf, MODELS / "voice_rf.joblib")
        info["voice_classifier"] = {"cv_accuracy": round(acc, 3)} | cm
        (MODELS / "voice_rf_metrics.json").write_text(json.dumps(info["voice_classifier"], ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    if force or not (MODELS / "squat_reference.json").exists():
        build_squat_reference().save(MODELS / "squat_reference.json")
    if force or not (MODELS / "prescription_iforest.joblib").exists():
        joblib.dump(train_prescription_iforest(), MODELS / "prescription_iforest.joblib")
    return info


def make_audio_pipeline(transcriber_backend: str = "auto") -> AudioPipeline:
    ensure_models()
    ref = ReferenceProfile.load(MODELS / "voice_reference.json")
    return AudioPipeline(get_transcriber(transcriber_backend), VoiceAnalyzer(ref, MODELS / "voice_rf.joblib"))


def make_exercise_analyzer() -> ExerciseAnalyzer:
    ensure_models()
    return ExerciseAnalyzer(ExerciseReference.load(MODELS / "squat_reference.json"))


def make_prescription_analyzer() -> PrescriptionAnalyzer:
    ensure_models()
    model, thr = joblib.load(MODELS / "prescription_iforest.joblib")
    return PrescriptionAnalyzer(model, thr)


def make_vitals_detector(settings: Settings) -> StreamingVitalsDetector:
    return StreamingVitalsDetector(settings.vitals)


def load_model_path(name: str) -> Path:
    return MODELS / name
