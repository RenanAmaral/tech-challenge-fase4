"""Detecção de alterações vocais: fadiga, disartria e dispneia.

Duas camadas complementares:

1. **Desvio do normal (explicável)** — cada feature é convertida em z-score
   contra um perfil de referência (falantes saudáveis ou o baseline do próprio
   paciente). Cada condição soma os desvios na direção clinicamente esperada
   (ex.: disartria ↑jitter, ↑shimmer, ↓HNR, ↑irregularidade de ritmo).
2. **Classificador supervisionado (opcional)** — Random Forest sobre as mesmas
   features, treinado com ``scripts/train_voice_model.py``. Usado para o
   diagnóstico diferencial quando há um modelo salvo em ``models/``.

O escore de anomalia do evento é o maior escore de desvio entre as condições.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .acoustic import FEATURE_NAMES, AcousticFeatures

# direção esperada (+1 = aumenta na condição) e peso de cada feature
CONDITION_SIGNATURES: dict[str, dict[str, tuple[int, float]]] = {
    "fadiga": {
        "mean_pause_s": (+1, 1.0),
        "pause_ratio": (+1, 0.8),
        "syllable_rate": (-1, 1.0),
        "intensity_db": (-1, 0.8),
        "hnr_db": (-1, 0.5),
    },
    "disartria": {
        "jitter_pct": (+1, 1.0),
        "shimmer_pct": (+1, 1.0),
        "hnr_db": (-1, 0.8),
        "rate_cv": (+1, 0.6),
        "syllable_rate": (-1, 0.5),
    },
    "dispneia": {
        "breath_per_min": (+1, 1.2),
        "syllables_per_phrase": (-1, 1.0),
        "pause_ratio": (+1, 0.6),
    },
}

# piso absoluto do desvio-padrão por feature (unidades da feature)
ABS_STD_FLOOR = {"f0_sd_st": 0.3, "jitter_pct": 0.2, "shimmer_pct": 1.0, "hnr_db": 2.0, "intensity_db": 3.0,
                 "syllable_rate": 0.4, "rate_cv": 0.05, "pause_ratio": 0.05, "mean_pause_s": 0.1,
                 "syllables_per_phrase": 1.0, "breath_per_min": 2.0}

CONDITION_LABELS = {
    "fadiga": "sinais de fadiga vocal (fala lenta, pausas longas, voz fraca)",
    "disartria": "sinais compatíveis com disartria (instabilidade vocal, ritmo irregular)",
    "dispneia": "sinais de dificuldade respiratória (frases curtas, inspirações audíveis)",
}


@dataclass
class ReferenceProfile:
    mean: dict[str, float]
    std: dict[str, float]
    n: int = 0
    source: str = "synthetic-healthy"

    @classmethod
    def from_features(cls, feats: list[AcousticFeatures], source: str = "healthy") -> "ReferenceProfile":
        X = np.array([f.vector(FEATURE_NAMES) for f in feats])
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=1) if len(X) > 1 else np.ones_like(mean)
        # piso no desvio-padrão: evita z-scores explosivos com referência pequena
        floor = np.array([ABS_STD_FLOOR[n] for n in FEATURE_NAMES])
        std = np.maximum(std, np.maximum(0.15 * np.abs(mean), floor))
        return cls(dict(zip(FEATURE_NAMES, mean.round(5))), dict(zip(FEATURE_NAMES, std.round(5))), len(X), source)

    def z(self, f: AcousticFeatures) -> dict[str, float]:
        v = dict(zip(FEATURE_NAMES, f.vector(FEATURE_NAMES)))
        return {k: (v[k] - self.mean[k]) / self.std[k] for k in FEATURE_NAMES}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceProfile":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class VoiceAssessment:
    scores: dict[str, float]
    main_condition: str | None
    anomaly_score: float
    z_scores: dict[str, float]
    drivers: list[str] = field(default_factory=list)
    classifier_probs: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return {
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "main_condition": self.main_condition,
            "anomaly_score": round(self.anomaly_score, 3),
            "z_scores": {k: round(v, 2) for k, v in self.z_scores.items()},
            "drivers": self.drivers,
            "classifier_probs": self.classifier_probs,
        }


class VoiceAnalyzer:
    def __init__(self, reference: ReferenceProfile, model_path: str | Path | None = None, threshold: float = 0.5):
        self.reference = reference
        self.threshold = threshold
        self.model = None
        if model_path and Path(model_path).exists():
            import joblib

            self.model = joblib.load(model_path)

    @staticmethod
    def _to_score(mean_z: float) -> float:
        # z médio 0 → 0; z ≈ 2 → 0,55; z ≈ 4 → 0,80; satura em 1
        return float(1 - np.exp(-max(0.0, mean_z) / 2.5))

    def assess(self, feats: AcousticFeatures) -> VoiceAssessment:
        z = self.reference.z(feats)
        scores: dict[str, float] = {}
        drivers: list[str] = []
        for cond, sig in CONDITION_SIGNATURES.items():
            total = sum(w * np.clip(d * z[f], 0, 8) for f, (d, w) in sig.items())
            scores[cond] = self._to_score(total / sum(w for _, w in sig.values()))
        probs = None
        if self.model is not None:
            p = self.model.predict_proba(feats.vector(FEATURE_NAMES).reshape(1, -1))[0]
            probs = {c: round(float(pi), 3) for c, pi in zip(self.model.classes_, p)}
        main = max(scores, key=scores.get)
        if probs:  # diferencial pelo classificador quando disponível
            best = max(probs, key=probs.get)
            main = best if best != "normal" else main
        anomaly = max(scores.values())
        if probs and probs.get("normal", 0) > 0.8:
            anomaly *= 0.5  # classificador confiante em voz normal atenua o alerta
        if anomaly < self.threshold:
            main = None
        else:
            for f, (d, _) in CONDITION_SIGNATURES[main].items():
                if d * z[f] >= 2:
                    drivers.append(f"{f}: z={z[f]:+.1f}")
        return VoiceAssessment(scores, main, anomaly, z, drivers, probs)
