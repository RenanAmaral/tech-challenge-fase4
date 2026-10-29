"""Detecção de anomalias em séries temporais de sinais vitais.

Três detectores combinados (o maior escore vence), todos em modo *streaming*
— cada amostra é avaliada no momento em que chega, como num monitor à beira
do leito:

1. **Limiares clínicos** (inspirados no NEWS2): SpO2, FC, PA sistólica, FR, T.
2. **Desvio do baseline do paciente**: z-score robusto (mediana/MAD) contra
   uma janela das últimas horas. O baseline *não aprende* com amostras
   anômalas, para não normalizar uma piora lenta.
3. **Multivariado**: soma dos z² dos sinais (distância de Mahalanobis com
   covariância diagonal) → p-valor qui-quadrado. Captura a *deterioração
   silenciosa*: vários sinais piorando juntos sem que nenhum cruze limiar.

Um ``IsolationForestVitals`` (scikit-learn) é oferecido como detector
estatístico offline para comparação no relatório.

Amostras anômalas consecutivas formam episódios; cada episódio gera um
``Event`` ao ser confirmado e outro se escalar para crítico.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import chi2

from ..config import VitalsDetectorConfig
from ..events import Event, Modality

SIGNALS = ["hr", "spo2", "sbp", "rr", "temp"]
LABELS_PT = {"hr": "FC", "spo2": "SpO2", "sbp": "PAS", "rr": "FR", "temp": "Temperatura"}
UNITS = {"hr": "bpm", "spo2": "%", "sbp": "mmHg", "rr": "irpm", "temp": "°C"}
THRESHOLD_TEXT = {"dessaturacao": "SpO2 abaixo do limiar", "taquicardia": "FC acima do limiar",
                  "bradicardia": "FC abaixo do limiar", "hipotensao": "PAS abaixo do limiar",
                  "hipertensao": "PAS acima do limiar", "taquipneia": "FR acima do limiar",
                  "bradipneia": "FR abaixo do limiar", "febre": "temperatura acima do limiar"}
_SIG_OF = {"dessaturacao": "spo2", "taquicardia": "hr", "bradicardia": "hr", "hipotensao": "sbp",
           "hipertensao": "sbp", "taquipneia": "rr", "bradipneia": "rr", "febre": "temp"}
# z-score por sinal e direção → tipo
_Z_KIND = {("hr", 1): "taquicardia", ("hr", -1): "bradicardia", ("spo2", -1): "dessaturacao",
           ("spo2", 1): "sinais_vitais_anomalos", ("sbp", -1): "hipotensao", ("sbp", 1): "hipertensao",
           ("rr", 1): "taquipneia", ("rr", -1): "bradipneia", ("temp", 1): "febre", ("temp", -1): "hipotermia"}
Z_FLOOR = {"hr": 3.0, "spo2": 0.6, "sbp": 5.0, "rr": 1.0, "temp": 0.3}
# faixa "sem alerta" (≈ pontuação 0 do NEWS2): desvio do baseline só conta fora dela
SOFT_BAND = {"hr": (55, 100), "spo2": (95, 101), "sbp": (105, 160), "rr": (10, 20), "temp": (35.5, 37.8)}
KIND_PT = {"dessaturacao": "dessaturação", "taquicardia": "taquicardia", "bradicardia": "bradicardia",
           "hipotensao": "hipotensão", "hipertensao": "hipertensão", "taquipneia": "taquipneia",
           "bradipneia": "bradipneia", "febre": "febre", "hipotermia": "hipotermia",
           "deterioracao_silenciosa": "deterioração combinada de vários sinais",
           "sinais_vitais_anomalos": "padrão anômalo"}


def _fmt(kind: str, s: dict) -> str:
    k = _SIG_OF[kind]
    return f"{LABELS_PT[k]} {s[k]:g} {UNITS[k]}"


def _beyond(value: float, warn: float, crit: float, lower: bool) -> float:
    """Escore 0–1 para valor além do limiar (``lower``: anormal quando baixo)."""
    if lower:
        if value >= warn:
            return 0.0
        frac = (warn - value) / max(1e-9, warn - crit)
    else:
        if value <= warn:
            return 0.0
        frac = (value - warn) / max(1e-9, crit - warn)
    return float(min(1.0, 0.55 + 0.3 * min(frac, 1.0) + 0.1 * max(0.0, frac - 1.0)))


@dataclass
class SampleScore:
    timestamp: datetime
    score: float
    threshold_score: float
    z_score: float
    multi_score: float
    reasons: list[str]
    kind: str = ""


@dataclass
class Episode:
    start: datetime
    end: datetime
    peak_score: float
    reasons: list[str]
    peak_values: dict[str, float]
    n_samples: int = 0
    kind: str = ""


@dataclass
class _PatientState:
    buf: dict[str, deque] = field(default_factory=dict)
    run: int = 0
    calm: int = 0
    episode: Episode | None = None
    emitted_level: int = -1
    last_emit: datetime | None = None


class StreamingVitalsDetector:
    def __init__(self, cfg: VitalsDetectorConfig | None = None, baseline_samples: int = 240, warmup: int = 30,
                 refresh_minutes: float = 10.0):
        self.cfg = cfg or VitalsDetectorConfig()
        self.refresh = refresh_minutes  # reemite episódio em curso (mantém a fusão informada)
        self.baseline_samples = baseline_samples
        self.warmup = warmup
        self.state: dict[str, _PatientState] = {}
        self.closed: dict[str, list[Episode]] = {}

    # ------------------------------------------------------------------ escore
    def _thresholds(self, s: dict) -> tuple[float, list[str], dict[str, float]]:
        L = self.cfg.limits
        parts = {
            "dessaturacao": _beyond(s["spo2"], L.spo2_low, L.spo2_critical, True),
            "taquicardia": _beyond(s["hr"], L.hr_high, L.hr_critical_high, False),
            "bradicardia": _beyond(s["hr"], L.hr_low, L.hr_low - 10, True),
            "hipotensao": _beyond(s["sbp"], L.sbp_low + 10, L.sbp_low, True),
            "hipertensao": _beyond(s["sbp"], L.sbp_high, L.sbp_high + 30, False),
            "taquipneia": _beyond(s["rr"], L.rr_high, L.rr_high + 6, False),
            "bradipneia": _beyond(s["rr"], L.rr_low, L.rr_low - 3, True),
            "febre": _beyond(s.get("temp", 36.8), L.temp_high, L.temp_high + 1.5, False),
        }
        reasons = [f"{THRESHOLD_TEXT[k]} ({_fmt(k, s)})" for k, v in parts.items() if v > 0]
        return max(parts.values()), reasons, {k: v for k, v in parts.items() if v > 0}

    def score_sample(self, patient_id: str, s: dict) -> SampleScore:
        st = self.state.setdefault(patient_id, _PatientState(
            buf={k: deque(maxlen=self.baseline_samples) for k in SIGNALS}))
        th_score, reasons, th_parts = self._thresholds(s)
        z_score = multi = 0.0
        z_kind = ""
        n = len(st.buf["hr"])
        if n >= self.warmup:
            zs = {}
            for k in SIGNALS:
                arr = np.fromiter(st.buf[k], float)
                med = np.median(arr)
                mad = 1.4826 * np.median(np.abs(arr - med))
                zs[k] = (s[k] - med) / max(mad, Z_FLOOR[k])
            lo_hi = {k: (s[k] < SOFT_BAND[k][0], s[k] > SOFT_BAND[k][1]) for k in SIGNALS}
            eligible = {k: z for k, z in zs.items() if (z < 0 and lo_hi[k][0]) or (z > 0 and lo_hi[k][1])}
            zmax_k = max(eligible, key=lambda k: abs(eligible[k])) if eligible else None
            zmax = abs(zs[zmax_k]) if zmax_k else 0.0
            if zmax_k and zmax > self.cfg.z_threshold:
                z_score = float(min(0.75, 0.5 + 0.05 * (zmax - self.cfg.z_threshold)))
                reasons.append(f"{LABELS_PT[zmax_k]} desviou {zs[zmax_k]:+.1f}σ do baseline do paciente")
                z_kind = _Z_KIND[(zmax_k, 1 if zs[zmax_k] > 0 else -1)]
            # direção clinicamente ruim: SpO2 e PAS baixas, FC/FR/T altas
            bad = {"hr": zs["hr"], "spo2": -zs["spo2"], "sbp": -zs["sbp"], "rr": zs["rr"], "temp": zs["temp"]}
            d2 = sum(max(0.0, v) ** 2 for v in bad.values())
            p = chi2.sf(d2, df=len(SIGNALS))
            n_bad = sum(v > 2.0 for v in bad.values())
            if p < 1e-6 and n_bad >= 3:
                multi = float(min(0.8, 0.55 + 0.04 * (-np.log10(p) - 6)))
                reasons.append(f"deterioração combinada ({n_bad} sinais, p={p:.1e})")
        score = max(th_score, z_score, multi)
        if th_parts:  # limiar clínico cruzado define o tipo
            kind = max(th_parts, key=th_parts.get)
        elif multi >= z_score and multi > 0:
            kind = "deterioracao_silenciosa"
        else:
            kind = z_kind
        # baseline só aprende com amostras normais
        if score < 0.5:
            for k in SIGNALS:
                st.buf[k].append(float(s[k]))
        return SampleScore(s["timestamp"], score, th_score, z_score, multi, reasons, kind)

    # -------------------------------------------------------------- episódios
    def update(self, patient_id: str, s: dict) -> tuple[SampleScore, list[Event]]:
        """Processa uma amostra; devolve o escore e eventos novos (se houver)."""
        sc = self.score_sample(patient_id, s)
        st = self.state[patient_id]
        events: list[Event] = []
        if sc.score >= 0.5:
            st.run += 1
            st.calm = 0
            if st.episode is None and st.run >= self.cfg.min_episode_samples:
                st.episode = Episode(sc.timestamp, sc.timestamp, sc.score, list(sc.reasons),
                                     {k: s[k] for k in SIGNALS}, 1, sc.kind)
            elif st.episode is not None:
                ep = st.episode
                ep.end = sc.timestamp
                ep.n_samples += 1
                for r in sc.reasons:
                    if r not in ep.reasons and len(ep.reasons) < 6:
                        ep.reasons.append(r)
                if sc.score > ep.peak_score:
                    ep.peak_score = sc.score
                    ep.peak_values = {k: s[k] for k in SIGNALS}
                    ep.kind = sc.kind or ep.kind
            if st.episode is not None:
                level = 2 if st.episode.peak_score >= 0.8 else 1
                stale = st.last_emit is not None and (sc.timestamp - st.last_emit).total_seconds() >= self.refresh * 60
                if level > st.emitted_level or stale:
                    st.emitted_level = max(level, st.emitted_level)
                    st.last_emit = sc.timestamp
                    events.append(self._event(patient_id, st.episode, sc.timestamp, ongoing=stale))
        else:
            st.run = 0
            st.calm += 1
            if st.episode is not None and st.calm >= 5:
                self.closed.setdefault(patient_id, []).append(st.episode)
                st.episode = None
                st.emitted_level = -1
                st.last_emit = None
        return sc, events

    def _event(self, patient_id: str, ep: Episode, ts: datetime, ongoing: bool = False) -> Event:
        vals = ", ".join(f"{LABELS_PT[k]} {ep.peak_values[k]:g} {UNITS[k]}" for k in SIGNALS)
        kind = ep.kind or "sinais_vitais_anomalos"
        mins = int((ts - ep.start).total_seconds() // 60)
        when = f" — em curso há {mins} min" if ongoing else ""
        return Event(
            patient_id=patient_id, modality=Modality.VITALS, kind=kind, anomaly_score=ep.peak_score,
            timestamp=ts, source="anomaly.vitals",
            description=f"Sinais vitais: {KIND_PT.get(kind, kind)}{when}. Pior valor: {vals}.",
            evidence={"reasons": ep.reasons, "values": ep.peak_values, "episode_start": ep.start.isoformat(),
                      "ongoing": ongoing},
        )

    # ------------------------------------------------------------------ batch
    def process_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[Event]]:
        """Replay de um DataFrame amostra a amostra (simula o tempo real)."""
        rows, events = [], []
        for rec in df.to_dict("records"):
            sc, evs = self.update(rec["patient_id"], rec)
            events += evs
            rows.append({"score": sc.score, "threshold_score": sc.threshold_score, "z_score": sc.z_score,
                         "multi_score": sc.multi_score, "reasons": "; ".join(sc.reasons)})
        out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        out["flag"] = _confirm(out["score"].to_numpy() >= 0.5, self.cfg.min_episode_samples)
        return out, events


def _confirm(mask: np.ndarray, k: int) -> np.ndarray:
    """Mantém apenas sequências com pelo menos ``k`` amostras consecutivas."""
    out = np.zeros_like(mask)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= k:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


class IsolationForestVitals:
    """Detector não supervisionado (offline) para comparação."""

    def __init__(self, contamination: float = 0.02, seed: int = 0):
        from sklearn.ensemble import IsolationForest

        self.model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)

    @staticmethod
    def features(df: pd.DataFrame) -> np.ndarray:
        X = df[SIGNALS].astype(float)
        d = X.diff(15).fillna(0)
        return np.hstack([X.to_numpy(), d.to_numpy()])

    def fit(self, df: pd.DataFrame) -> "IsolationForestVitals":
        self.model.fit(self.features(df))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.features(df)) == -1
