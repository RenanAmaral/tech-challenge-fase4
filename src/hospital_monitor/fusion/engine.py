"""Fusão multimodal tardia (late fusion) → escore de risco por paciente.

Para cada paciente, mantém os eventos da janela deslizante (padrão 15 min,
pelo relógio dos próprios eventos, o que permite replay determinístico).
Em cada modalidade vale o evento mais grave da janela; as modalidades são
combinadas por **noisy-OR ponderado**:

    risco = 1 − Π_m (1 − w_m · s_m)

Isso preserva a gravidade de um único sinal forte (uma dessaturação crítica
sozinha já gera alerta crítico) e faz o risco crescer quando várias
modalidades apontam problema. *Eventos sentinela* (queda, gaze retida) não
são atenuados pelo peso da modalidade. Há ainda um reforço explícito por
**concordância**: cada modalidade adicional com s ≥ 0,5 soma ``agreement_boost``.

Comparada à média ponderada (plano inicial), o noisy-OR evita que modalidades
normais "diluam" um achado grave — propriedade desejável em segurança do
paciente.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from ..config import FusionConfig
from ..events import Event, Severity


@dataclass
class RiskAssessment:
    patient_id: str
    timestamp: datetime
    risk: float
    level: Severity
    contributions: dict[str, float]
    agreeing: list[str]
    events: list[Event] = field(default_factory=list)
    trigger_event_id: str = ""

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp.isoformat(),
            "risk": round(self.risk, 3),
            "level": self.level.value,
            "contributions": {k: round(v, 3) for k, v in self.contributions.items()},
            "agreeing_modalities": self.agreeing,
            "event_ids": [e.event_id for e in self.events],
            "trigger_event_id": self.trigger_event_id,
        }


class RiskFusionEngine:
    def __init__(self, cfg: FusionConfig | None = None):
        self.cfg = cfg or FusionConfig()
        self.window = timedelta(minutes=self.cfg.window_minutes)
        self._events: dict[str, deque[Event]] = defaultdict(deque)
        self.history: list[RiskAssessment] = []
        self._listeners: list[Callable[[RiskAssessment], None]] = []

    def on_assessment(self, fn: Callable[[RiskAssessment], None]) -> None:
        self._listeners.append(fn)

    def fuse(self, events: list[Event]) -> tuple[float, dict[str, float], list[str]]:
        best: dict[str, float] = {}
        for e in events:
            m = e.modality.value
            best[m] = max(best.get(m, 0.0), e.anomaly_score)
        prod = 1.0
        contrib = {}
        for m, s in best.items():
            w = self.cfg.weights.get(m, 0.5)
            contrib[m] = w * s
            prod *= 1 - w * s
        risk = 1 - prod
        # eventos sentinela (queda, material retido…) não são atenuados pelo peso da modalidade
        sentinel = [e.anomaly_score for e in events if e.kind in self.cfg.sentinel_kinds]
        if sentinel:
            risk = max(risk, max(sentinel))
        agreeing = sorted(m for m, s in best.items() if s >= 0.5)
        if len(agreeing) > 1:
            risk += self.cfg.agreement_boost * (len(agreeing) - 1)
        return min(1.0, risk), contrib, agreeing

    def ingest(self, event: Event, emit: bool = True) -> RiskAssessment:
        """Adiciona o evento à janela do paciente e recalcula o risco.

        ``emit=False`` atualiza o estado sem avisar os ouvintes (usado para
        eventos simultâneos do mesmo paciente, avaliados juntos no último).
        """
        q = self._events[event.patient_id]
        q.append(event)
        # mantém ordem temporal mesmo se eventos chegarem fora de ordem
        if len(q) > 1 and q[-2].timestamp > event.timestamp:
            items = sorted(q, key=lambda e: e.timestamp)
            q.clear()
            q.extend(items)
        now = max(e.timestamp for e in q)
        while q and q[0].timestamp < now - self.window:
            q.popleft()
        window_events = [e for e in q if e.timestamp <= now]
        risk, contrib, agreeing = self.fuse(window_events)
        level = Severity.from_score(risk, self.cfg.attention_threshold, self.cfg.critical_threshold)
        a = RiskAssessment(event.patient_id, now, risk, level, contrib, agreeing, list(window_events), event.event_id)
        self.history.append(a)
        if emit:
            for fn in self._listeners:
                fn(a)
        return a

    def current_risk(self, patient_id: str) -> RiskAssessment | None:
        for a in reversed(self.history):
            if a.patient_id == patient_id:
                return a
        return None
