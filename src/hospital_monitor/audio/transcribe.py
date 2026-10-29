"""Transcrição de fala (speech-to-text).

* ``WhisperTranscriber`` — faster-whisper (modelo Whisper local, gratuito,
  pt-BR, timestamps por palavra). Baixa o modelo do Hugging Face na 1ª execução.
* ``SidecarTranscriber`` — lê ``<audio>.txt`` ou ``<audio>.json`` ao lado do
  arquivo. Usado em testes, CI e quando não há modelo disponível offline.

``get_transcriber("auto")`` tenta Whisper e cai para o sidecar.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    language: str = "pt"
    segments: list[dict] = field(default_factory=list)  # [{start, end, text}]
    backend: str = ""


class Transcriber(Protocol):
    def transcribe(self, audio_path: str | Path) -> Transcript: ...


class SidecarTranscriber:
    name = "sidecar"

    def transcribe(self, audio_path: str | Path) -> Transcript:
        p = Path(audio_path)
        js, txt = p.with_suffix(".json"), p.with_suffix(".txt")
        if js.exists():
            d = json.loads(js.read_text(encoding="utf-8"))
            return Transcript(d["text"], d.get("language", "pt"), d.get("segments", []), "sidecar")
        if txt.exists():
            return Transcript(txt.read_text(encoding="utf-8").strip(), backend="sidecar")
        raise FileNotFoundError(f"sem transcrição para {p.name} (esperado {txt.name} ou {js.name})")


class WhisperTranscriber:  # pragma: no cover - exige download do modelo
    name = "whisper"

    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str | Path) -> Transcript:
        segs, info = self.model.transcribe(str(audio_path), language="pt", word_timestamps=True, vad_filter=True)
        segments = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segs]
        return Transcript(" ".join(s["text"] for s in segments), info.language, segments, "faster-whisper")


def get_transcriber(backend: str = "auto", model_size: str = "small") -> Transcriber:
    if backend in ("auto", "whisper"):
        try:
            return WhisperTranscriber(model_size)
        except Exception as exc:  # sem pacote ou sem acesso ao modelo
            if backend == "whisper":
                raise
            log.warning("Whisper indisponível (%s); usando transcrição sidecar.", exc)
    return SidecarTranscriber()
