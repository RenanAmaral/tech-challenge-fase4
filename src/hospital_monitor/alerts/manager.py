"""Gestão de alertas: limiar, deduplicação (cooldown), escalonamento e reconhecimento.

Política anti–fadiga de alertas:
  * só gera alerta a partir do nível ``min_level`` (padrão: atenção);
  * dentro do ``cooldown`` (padrão 30 min) não repete alerta de mesmo nível ou
    inferior para o mesmo paciente — os eventos novos são *anexados* ao alerta
    aberto;
  * evidência de uma modalidade nova é notificada como *atualização* do alerta;
  * escalonamento (atenção → crítico) sempre notifica imediatamente e marca o
    alerta anterior como ``escalated``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..config import AlertConfig
from ..events import Severity
from ..fusion.engine import RiskAssessment
from .notifiers import Notifier

MODALITY_PT = {"vitals": "sinais vitais", "audio": "voz", "text": "relato (NLP)", "video": "vídeo",
               "prescription": "prescrição", "movement": "movimentação"}


@dataclass
class Alert:
    alert_id: str
    patient_id: str
    level: str
    risk: float
    created_at: datetime
    title: str
    details: list[str]
    modalities: list[str]
    event_ids: list[str]
    status: str = "open"
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    updated_at: datetime | None = None
    history: list[dict] = field(default_factory=list)
    evidence: dict = field(default_factory=dict, repr=False)  # (modalidade, tipo) → (escore, descrição)

    def absorb(self, events) -> None:
        """Uma linha de detalhe por (modalidade, tipo): a mais grave e mais recente."""
        for e in sorted(events, key=lambda e: e.timestamp):
            k = f"{e.modality.value}:{e.kind}"
            if k not in self.evidence or e.anomaly_score >= self.evidence[k][0]:
                self.evidence[k] = (e.anomaly_score, e.description)
        self.details = [d for _, d in sorted(self.evidence.values(), key=lambda x: -x[0])]

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id, "patient_id": self.patient_id, "level": self.level,
            "risk": round(self.risk, 3), "created_at": self.created_at.isoformat(), "title": self.title,
            "details": self.details, "modalities": self.modalities, "event_ids": self.event_ids,
            "status": self.status, "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "history": sorted(self.history, key=lambda h: h["t"]),
        }


def _title(a: RiskAssessment) -> str:
    lvl = "crítico" if a.level == Severity.CRITICAL else "elevado"
    if len(a.agreeing) > 1:
        mods = ", ".join(MODALITY_PT.get(m, m) for m in a.agreeing)
        return f"Risco {lvl}: {mods} indicam piora ao mesmo tempo."
    top = max(a.contributions, key=a.contributions.get)
    return f"Risco {lvl} por achado em {MODALITY_PT.get(top, top)}."


class AlertManager:
    def __init__(self, cfg: AlertConfig | None = None, notifiers: list[Notifier] | None = None,
                 store_path: str | Path | None = None):
        self.cfg = cfg or AlertConfig()
        self.notifiers = notifiers or []
        self.alerts: dict[str, Alert] = {}
        self.suppressed = 0
        self.updates = 0
        self.store_path = Path(store_path) if store_path else None

    def _open_for(self, patient_id: str, now: datetime) -> Alert | None:
        cands = [a for a in self.alerts.values() if a.patient_id == patient_id
                 and now - (a.updated_at or a.created_at) <= timedelta(minutes=self.cfg.cooldown_minutes)]
        return max(cands, key=lambda a: a.created_at) if cands else None

    def handle(self, a: RiskAssessment) -> Alert | None:
        if a.level.rank < Severity(self.cfg.min_level).rank:
            return None
        recent = self._open_for(a.patient_id, a.timestamp)
        if recent and Severity(recent.level).rank >= a.level.rank:
            # deduplica: anexa evidências ao alerta aberto
            new_ids = [e.event_id for e in a.events if e.event_id not in recent.event_ids]
            new_mods = set(a.contributions) - set(recent.modalities)
            if new_ids:
                recent.event_ids += new_ids
                recent.absorb(a.events)
                recent.modalities = sorted(set(recent.modalities) | set(a.contributions))
                recent.risk = max(recent.risk, a.risk)
                recent.updated_at = a.timestamp
                if len(recent.modalities) > 1:
                    recent.title = f"Risco {'crítico' if recent.level == 'critical' else 'elevado'}: " + \
                        ", ".join(MODALITY_PT.get(m, m) for m in recent.modalities) + " indicam piora ao mesmo tempo."
                recent.history.append({"t": a.timestamp.isoformat(), "risk": round(a.risk, 3),
                                       "action": "nova modalidade: " + ", ".join(sorted(new_mods)) if new_mods
                                       else "anexado"})
            if new_mods:
                # evidência de outra modalidade é informação nova: notifica como atualização
                self.updates += 1
                self._notify(recent, update=True)
            else:
                self.suppressed += 1
            self._persist()
            return None
        if recent and recent.status == "open":
            recent.status = "escalated"
        alert = Alert(
            alert_id=uuid.uuid4().hex[:8], patient_id=a.patient_id, level=a.level.value, risk=a.risk,
            created_at=a.timestamp, title=_title(a), details=[], modalities=sorted(a.contributions),
            event_ids=[e.event_id for e in a.events], updated_at=a.timestamp,
            history=[{"t": a.timestamp.isoformat(), "risk": round(a.risk, 3),
                      "action": f"escalonado de {recent.alert_id}" if recent else "criado"}],
        )
        alert.absorb(a.events)
        self.alerts[alert.alert_id] = alert
        self._notify(alert)
        self._persist()
        return alert

    def _notify(self, alert: Alert, update: bool = False) -> None:
        import copy

        payload = copy.copy(alert)
        payload._is_update = update  # lido por format_alert
        for n in self.notifiers:
            n.send(payload)

    def acknowledge(self, alert_id: str, user: str, when: datetime | None = None) -> Alert:
        al = self.alerts[alert_id]
        al.status = "acknowledged"
        al.acknowledged_by = user
        al.acknowledged_at = when or datetime.now(timezone.utc)
        al.history.append({"t": al.acknowledged_at.isoformat(), "action": f"reconhecido por {user}"})
        self._persist()
        return al

    def list(self, status: str | None = None, patient_id: str | None = None) -> list[Alert]:
        out = [a for a in self.alerts.values() if (not status or a.status == status)
               and (not patient_id or a.patient_id == patient_id)]
        return sorted(out, key=lambda a: a.created_at)

    def _persist(self) -> None:
        if self.store_path:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            self.store_path.write_text(json.dumps([a.to_dict() for a in self.list()], ensure_ascii=False, indent=1),
                                       encoding="utf-8")
