"""Análise de vídeo cirúrgico com YOLOv8: detecção de objetos + regras de segurança.

O detector (``YoloDetector``) roda Ultralytics YOLOv8 com pesos treinados para
instrumental cirúrgico (ex.: fine-tuning em m2cai16-tool / Cholec80 ou dataset
próprio rotulado no Roboflow). A camada de regras é independente do modelo e
consome detecções por frame no formato ``{"cls", "conf", "bbox": [x1,y1,x2,y2]}``
(coordenadas normalizadas 0–1).

Regras implementadas:
  * instrumento fora do campo estéril por tempo prolongado;
  * campo visual obstruído (nenhuma detecção) por vários segundos;
  * contagem de gazes/instrumentos no fim menor que no início
    (possível retenção de material — evento crítico).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ..events import Event, Modality

COUNTED = ("gaze",)
INSTRUMENTS = ("pinca", "tesoura", "bisturi", "porta_agulha")


class YoloDetector:  # pragma: no cover - exige pesos
    def __init__(self, weights: str = "models/surgical_yolov8n.pt", conf: float = 0.4,
                 class_map: dict[str, str] | None = None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.class_map = class_map or {}

    def detect_video(self, path: str | Path, fps: float = 5.0) -> list[list[dict]]:
        from .pose import read_video_frames

        out = []
        for _, _, frame in read_video_frames(path, fps):
            h, w = frame.shape[:2]
            r = self.model(frame, conf=self.conf, verbose=False)[0]
            dets = []
            for b in r.boxes:
                name = r.names[int(b.cls)]
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                dets.append({"cls": self.class_map.get(name, name), "conf": round(float(b.conf), 3),
                             "bbox": [x1 / w, y1 / h, x2 / w, y2 / h]})
            out.append(dets)
        return out


@dataclass
class SurgicalFinding:
    kind: str
    start_s: float
    end_s: float
    score: float
    description: str
    details: dict = field(default_factory=dict)


@dataclass
class SurgicalResult:
    patient_id: str
    fps: float
    n_frames: int
    findings: list[SurgicalFinding]
    counts_start: dict[str, int]
    counts_end: dict[str, int]
    counts_series: dict[str, list[int]]
    events: list[Event]

    def summary(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "duration_s": round(self.n_frames / self.fps, 1),
            "counts_start": self.counts_start,
            "counts_end": self.counts_end,
            "findings": [asdict(f) for f in self.findings],
            "events": [e.to_dict() for e in self.events],
        }


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _runs(mask: np.ndarray, max_gap: int = 0) -> list[tuple[int, int]]:
    """Intervalos [s, e] verdadeiros, unindo lacunas de até ``max_gap`` frames."""
    idx = np.where(mask)[0]
    if not len(idx):
        return []
    runs, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            runs.append((s, prev))
            s = i
        prev = i
    runs.append((s, prev))
    return runs


class SurgicalVideoAnalyzer:
    def __init__(self, sterile_field=(0.15, 0.15, 0.85, 0.85), outside_min_s: float = 10.0,
                 occlusion_min_s: float = 5.0, count_window_s: float = 30.0):
        self.field = sterile_field
        self.outside_min_s = outside_min_s
        self.occlusion_min_s = occlusion_min_s
        self.count_window_s = count_window_s

    def _inside(self, b) -> bool:
        cx, cy = _center(b)
        x1, y1, x2, y2 = self.field
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def analyze(self, detections: list[list[dict]], fps: float, patient_id: str, start_time: datetime) -> SurgicalResult:
        n = len(detections)
        findings: list[SurgicalFinding] = []
        classes = sorted({d["cls"] for f in detections for d in f})
        counts = {c: np.array([sum(1 for d in f if d["cls"] == c) for f in detections]) for c in classes}

        # 1) instrumento fora do campo estéril
        for inst in INSTRUMENTS:
            outside = np.array([any(d["cls"] == inst and not self._inside(d["bbox"]) for d in f) for f in detections])
            for s, e in _runs(outside, max_gap=int(2 * fps)):
                dur = (e - s + 1) / fps
                if dur >= self.outside_min_s:
                    sc = min(0.85, 0.55 + 0.3 * dur / 60)
                    findings.append(SurgicalFinding("instrument_outside_field", s / fps, e / fps, sc,
                                                    f"Instrumento '{inst}' fora do campo estéril por {dur:.0f} s.",
                                                    {"instrument": inst, "duration_s": round(dur, 1)}))

        # 2) obstrução da câmera / campo sem visibilidade
        empty = np.array([len(f) == 0 for f in detections])
        for s, e in _runs(empty):
            dur = (e - s + 1) / fps
            if dur >= self.occlusion_min_s:
                findings.append(SurgicalFinding("camera_occlusion", s / fps, e / fps, 0.55,
                                                f"Campo cirúrgico sem visibilidade por {dur:.0f} s.",
                                                {"duration_s": round(dur, 1)}))

        # 3) contagem inicial x final (mediana em janela, robusta a falhas de detecção)
        w = max(1, int(self.count_window_s * fps))
        def robust(c, sl):  # noqa: E306
            x = counts[c][sl]
            x = x[x > 0] if (x > 0).any() else x
            return int(round(np.percentile(x, 90))) if len(x) else 0
        start_c = {c: robust(c, slice(0, w)) for c in classes}
        end_c = {c: robust(c, slice(n - w, n)) for c in classes}
        for c in classes:
            if c == "mao":
                continue
            missing = start_c[c] - end_c[c]
            if missing > 0:
                crit = c in COUNTED
                findings.append(SurgicalFinding(
                    "gauze_count_mismatch" if crit else "instrument_count_mismatch", (n - 1) / fps, (n - 1) / fps,
                    0.95 if crit else 0.75,
                    f"Contagem de '{c}': {start_c[c]} no início e {end_c[c]} no fim — "
                    f"{missing} item(ns) não localizado(s). Possível retenção de material.",
                    {"class": c, "start": start_c[c], "end": end_c[c]}))

        events = [Event(patient_id=patient_id, modality=Modality.VIDEO, kind=f.kind, anomaly_score=f.score,
                        timestamp=start_time + timedelta(seconds=f.start_s), source="video.surgical",
                        description=f.description, evidence=f.details | {"t_s": round(f.start_s, 1)})
                  for f in findings]
        series = {c: counts[c].tolist() for c in classes}
        return SurgicalResult(patient_id, fps, n, findings, start_c, end_c, series, events)
