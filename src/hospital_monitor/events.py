"""Contrato de evento comum a todos os pipelines.

Cada módulo (vídeo, áudio, sinais vitais, prescrições, movimentação) publica
``Event`` no barramento. O motor de fusão consome apenas este formato, o que
mantém os pipelines desacoplados e testáveis de forma isolada.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Modality(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    VITALS = "vitals"
    PRESCRIPTION = "prescription"
    MOVEMENT = "movement"


class Severity(str, Enum):
    INFO = "info"
    ATTENTION = "attention"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float, attention: float = 0.5, critical: float = 0.8) -> "Severity":
        if score >= critical:
            return cls.CRITICAL
        if score >= attention:
            return cls.ATTENTION
        return cls.INFO

    @property
    def rank(self) -> int:
        return {"info": 0, "attention": 1, "critical": 2}[self.value]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


@dataclass
class Event:
    """Evento de anomalia emitido por um pipeline.

    Attributes:
        patient_id: identificador (pseudonimizado) do paciente.
        modality: tipo de dado de origem.
        kind: tipo de anomalia, ex. ``spo2_low``, ``exercise_deviation``.
        anomaly_score: 0 (normal) a 1 (anomalia certa).
        timestamp: instante em que a anomalia ocorreu.
        description: texto curto em pt-BR para a equipe médica.
        evidence: dados que sustentam o achado (valores, trechos, frames).
        source: módulo que gerou o evento.
    """

    patient_id: str
    modality: Modality
    kind: str
    anomaly_score: float
    timestamp: datetime = field(default_factory=utcnow)
    description: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        self.modality = Modality(self.modality)
        self.anomaly_score = float(min(1.0, max(0.0, self.anomaly_score)))
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    @property
    def severity(self) -> Severity:
        return Severity.from_score(self.anomaly_score)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["modality"] = self.modality.value
        d["timestamp"] = _iso(self.timestamp)
        d["severity"] = self.severity.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=_json_default)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        d = dict(d)
        d.pop("severity", None)
        ts = d.get("timestamp")
        if isinstance(ts, str):
            d["timestamp"] = datetime.fromisoformat(ts)
        return cls(**d)


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return _iso(o)
    if isinstance(o, Enum):
        return o.value
    try:  # numpy
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)
