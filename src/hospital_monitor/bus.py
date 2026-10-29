"""Barramento de eventos (pub/sub).

``InMemoryBus`` atende a demo e os testes. ``RedisStreamBus`` usa Redis Streams
quando a solução roda distribuída (docker compose). Ambos expõem a mesma
interface ``publish`` / ``subscribe``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Callable, Protocol

from .events import Event

log = logging.getLogger(__name__)

Handler = Callable[[Event], None]
TOPIC_ANOMALIES = "anomalies"


class EventBus(Protocol):
    def publish(self, event: Event, topic: str = TOPIC_ANOMALIES) -> None: ...

    def subscribe(self, handler: Handler, topic: str = TOPIC_ANOMALIES) -> None: ...


class InMemoryBus:
    """Barramento síncrono em memória. Mantém histórico para auditoria."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self.history: list[Event] = []

    def publish(self, event: Event, topic: str = TOPIC_ANOMALIES) -> None:
        self.history.append(event)
        for handler in list(self._subs[topic]):
            try:
                handler(event)
            except Exception:  # um consumidor com erro não derruba os demais
                log.exception("handler falhou para evento %s", event.event_id)

    def subscribe(self, handler: Handler, topic: str = TOPIC_ANOMALIES) -> None:
        self._subs[topic].append(handler)


class RedisStreamBus:  # pragma: no cover - exige Redis rodando
    """Barramento sobre Redis Streams (``XADD`` / ``XREAD``)."""

    def __init__(self, url: str = "redis://localhost:6379/0", stream_prefix: str = "hm:") -> None:
        import redis  # import tardio: dependência opcional

        self.r = redis.Redis.from_url(url, decode_responses=True)
        self.prefix = stream_prefix
        self._subs: dict[str, list[Handler]] = defaultdict(list)

    def publish(self, event: Event, topic: str = TOPIC_ANOMALIES) -> None:
        self.r.xadd(self.prefix + topic, {"data": event.to_json()}, maxlen=100_000)

    def subscribe(self, handler: Handler, topic: str = TOPIC_ANOMALIES) -> None:
        self._subs[topic].append(handler)

    def run_forever(self, topic: str = TOPIC_ANOMALIES, block_ms: int = 5000) -> None:
        last_id = "$"
        stream = self.prefix + topic
        while True:
            resp = self.r.xread({stream: last_id}, block=block_ms, count=100)
            for _, entries in resp or []:
                for entry_id, fields in entries:
                    last_id = entry_id
                    ev = Event.from_dict(json.loads(fields["data"]))
                    for h in self._subs[topic]:
                        try:
                            h(ev)
                        except Exception:
                            log.exception("handler falhou")
