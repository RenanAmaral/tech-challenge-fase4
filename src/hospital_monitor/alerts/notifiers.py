"""Canais de notificação da equipe médica (todos gratuitos).

* ``ConsoleNotifier`` — terminal (demo).
* ``FileNotifier`` — JSONL em ``outputs/alerts.jsonl`` (painel lê daqui).
* ``TelegramNotifier`` — Bot API; requer ``TELEGRAM_BOT_TOKEN`` e ``TELEGRAM_CHAT_ID``.
* ``DiscordNotifier`` — webhook; requer ``DISCORD_WEBHOOK_URL``.

``notifiers_from_env()`` ativa os canais externos só quando as variáveis
estão definidas (arquivo ``.env``).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

ICON = {"critical": "🔴", "attention": "🟠", "info": "⚪"}
LEVEL_PT = {"critical": "CRÍTICO", "attention": "ATENÇÃO", "info": "INFO"}


class Notifier(Protocol):
    def send(self, alert) -> None: ...


def format_alert(alert, markdown: bool = False) -> str:
    b = "*" if markdown else ""
    tag = "ATUALIZAÇÃO DE ALERTA" if getattr(alert, "_is_update", False) else "ALERTA"
    lines = [f"{ICON[alert.level]} {b}{tag} {LEVEL_PT[alert.level]}{b} — paciente {alert.patient_id} "
             f"(risco {alert.risk:.2f})", alert.title]
    for d in alert.details[:5]:
        lines.append(f"• {d}")
    lines.append(f"Modalidades: {', '.join(alert.modalities)} · {(alert.updated_at or alert.created_at):%d/%m %H:%M} · id {alert.alert_id}")
    return "\n".join(lines)


class ConsoleNotifier:
    def __init__(self, stream=None):
        import sys

        self.stream = stream or sys.stdout

    def send(self, alert) -> None:
        print("\n" + format_alert(alert) + "\n", file=self.stream)


class FileNotifier:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")


class TelegramNotifier:  # pragma: no cover - rede externa
    def __init__(self, token: str, chat_id: str):
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id

    def send(self, alert) -> None:
        import requests

        try:
            requests.post(self.url, json={"chat_id": self.chat_id, "text": format_alert(alert, True),
                                          "parse_mode": "Markdown"}, timeout=10).raise_for_status()
        except Exception as exc:
            log.error("Telegram falhou: %s", exc)


class DiscordNotifier:  # pragma: no cover - rede externa
    def __init__(self, webhook_url: str):
        self.url = webhook_url

    def send(self, alert) -> None:
        import requests

        try:
            requests.post(self.url, json={"content": format_alert(alert, True)}, timeout=10).raise_for_status()
        except Exception as exc:
            log.error("Discord falhou: %s", exc)


def notifiers_from_env() -> list[Notifier]:
    out: list[Notifier] = []
    if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        out.append(TelegramNotifier(os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]))
    if os.getenv("DISCORD_WEBHOOK_URL"):
        out.append(DiscordNotifier(os.environ["DISCORD_WEBHOOK_URL"]))
    return out
