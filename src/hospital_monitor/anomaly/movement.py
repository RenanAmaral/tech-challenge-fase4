"""Padrões de movimentação do paciente durante a internação.

Entrada: série de atividade (0–100) a cada 5 min, vinda de acelerômetro no
leito ou da *energia de movimento* do vídeo do quarto (``motion_energy``).

Anomalias:
  * ``imobilidade`` — atividade quase nula por ≥ 2 h acordado (risco de lesão
    por pressão, rebaixamento de consciência);
  * ``agitacao_noturna`` — atividade noturna muito acima do baseline noturno
    do paciente (risco de delirium e queda);
  * ``queda`` — pico abrupto seguido de imobilidade.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..events import Event, Modality
from .metrics import runs

TEXT = {
    "imobilidade": "Imobilidade prolongada no leito ({h:.1f} h sem movimentação) — avaliar consciência e risco de lesão por pressão.",
    "agitacao_noturna": "Agitação noturna ({h:.1f} h de atividade intensa) — risco de delirium e queda.",
    "queda": "Possível queda: pico de movimento seguido de imobilidade.",
}


def motion_energy(video_path: str | Path, step_s: float = 1.0) -> pd.DataFrame:  # pragma: no cover
    """Energia de movimento por diferença de frames (0–100), 1 valor/``step_s``."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps * step_s))
    prev, vals, i = None, [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (21, 21), 0)
            if prev is not None:
                vals.append(float((cv2.absdiff(g, prev) > 25).mean() * 100))
            prev = g
        i += 1
    cap.release()
    return pd.DataFrame({"t_s": np.arange(len(vals)) * step_s, "activity": vals})


class MovementAnalyzer:
    def __init__(self, step_min: int = 5, immobile_level: float = 3.0, immobile_hours: float = 2.0,
                 night=(22, 6), agitation_factor: float = 3.0, agitation_min_level: float = 25.0):
        self.step = step_min
        self.immobile_level = immobile_level
        self.immobile_n = int(immobile_hours * 60 / step_min)
        self.night = night
        self.agit_factor = agitation_factor
        self.agit_min = agitation_min_level

    def _is_night(self, ts: pd.Series) -> np.ndarray:
        h = ts.dt.hour.to_numpy()
        a, b = self.night
        return (h >= a) | (h < b)

    def analyze(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[Event]]:
        df = df.sort_values("timestamp").reset_index(drop=True)
        act = df["activity"].to_numpy(float)
        night = self._is_night(df["timestamp"])
        flag = np.zeros(len(df), dtype=object)
        flag[:] = ""
        events: list[Event] = []
        pid = df["patient_id"].iloc[0]

        # imobilidade acordado
        low = (act < self.immobile_level) & ~night
        for s, e in runs(low):
            if e - s + 1 >= self.immobile_n:
                h = (e - s + 1) * self.step / 60
                flag[s : e + 1] = "imobilidade"
                score = min(0.8, 0.6 + 0.05 * (h - 2))
                events.append(self._ev(pid, "imobilidade", score, df["timestamp"][s + self.immobile_n - 1], h, s, e, df))

        # agitação noturna: média móvel de 30 min vs baseline noturno robusto
        # média de 30 min só com amostras noturnas (evita a transição dia→noite)
        roll = pd.Series(np.where(night, act, np.nan)).rolling(6, min_periods=6).mean().to_numpy()
        base = np.median(act[night]) if night.any() else np.median(act)
        thr = max(self.agit_min, self.agit_factor * base)
        agit = night & (roll > thr)
        for s, e in runs(agit):
            if e - s + 1 >= 4:
                h = (e - s + 1) * self.step / 60
                flag[s : e + 1] = "agitacao_noturna"
                events.append(self._ev(pid, "agitacao_noturna", 0.7, df["timestamp"][s], h, s, e, df,
                                       {"baseline_noturno": round(float(base), 1), "limiar": round(float(thr), 1)}))

        # queda: pico > p99,5 e > 85, seguido de 3 amostras quase paradas
        p = max(85.0, float(np.percentile(act, 99.5)) * 0.95)
        for i in np.where(act >= p)[0]:
            nxt = act[i + 1 : i + 4]
            if len(nxt) == 3 and (nxt < 2.0).all():
                flag[i : i + 4] = "queda"
                events.append(self._ev(pid, "queda", 0.9, df["timestamp"][i], 0, i, i + 3, df))

        df["flag"] = flag
        df["pred"] = df["flag"] != ""
        return df, events

    def _ev(self, pid, kind, score, ts, h, s, e, df, extra=None) -> Event:
        return Event(
            patient_id=pid, modality=Modality.MOVEMENT, kind=kind, anomaly_score=score,
            timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts, source="anomaly.movement",
            description=TEXT[kind].format(h=h),
            evidence={"inicio": str(df["timestamp"][s]), "fim": str(df["timestamp"][e] + timedelta(minutes=self.step)),
                      "atividade_media": round(float(df["activity"][s : e + 1].mean()), 1)} | (extra or {}),
        )
