"""Orquestrador: liga barramento → fusão → alertas, com trilha de auditoria."""

from __future__ import annotations

import json
from pathlib import Path

from .alerts.manager import AlertManager
from .alerts.notifiers import Notifier
from .bus import InMemoryBus
from .config import Settings, load_settings
from .events import Event
from .fusion.engine import RiskFusionEngine


class Orchestrator:
    def __init__(self, settings: Settings | None = None, notifiers: list[Notifier] | None = None,
                 outputs_dir: str | Path | None = None):
        self.settings = settings or load_settings()
        self.out = Path(outputs_dir) if outputs_dir else None
        self.bus = InMemoryBus()
        self.fusion = RiskFusionEngine(self.settings.fusion)
        self.alerts = AlertManager(self.settings.alerts, notifiers,
                                   store_path=self.out / "alerts.json" if self.out else None)
        self.bus.subscribe(self._on_event)
        self.fusion.on_assessment(self.alerts.handle)

        self._hold = False

    def _on_event(self, event: Event) -> None:
        self.fusion.ingest(event, emit=not self._hold)

    def publish(self, event: Event) -> None:
        self.bus.publish(event)

    def publish_many(self, events: list[Event]) -> None:
        """Eventos simultâneos (ex.: voz + texto da mesma consulta) geram uma única avaliação."""
        for e in events[:-1]:
            self._hold = True
            try:
                self.publish(e)
            finally:
                self._hold = False
        if events:
            self.publish(events[-1])

    def replay(self, events: list[Event], pause_s: float = 0.0) -> None:
        """Publica eventos em ordem temporal (simula a chegada em tempo real)."""
        import time
        from itertools import groupby

        ordered = sorted(events, key=lambda e: (e.timestamp, e.patient_id))
        for _, grp in groupby(ordered, key=lambda e: (e.timestamp, e.patient_id)):
            self.publish_many(list(grp))
            if pause_s:
                time.sleep(pause_s)

    def dump(self) -> None:
        if not self.out:
            return
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.out / "events.jsonl", "w", encoding="utf-8") as fh:
            for e in self.bus.history:
                fh.write(e.to_json() + "\n")
        with open(self.out / "assessments.jsonl", "w", encoding="utf-8") as fh:
            for a in self.fusion.history:
                fh.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")
