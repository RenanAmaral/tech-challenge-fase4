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


def _read_text(p: Path) -> str:
    """UTF-8 (com ou sem BOM do Bloco de Notas); cai para cp1252 em arquivos antigos do Windows."""
    raw = p.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        log.warning("%s não está em UTF-8; lendo como cp1252.", p.name)
        return raw.decode("cp1252")


def whisper_works(model_size: str = "tiny", timeout_s: int = 300) -> bool:
    """Testa o Whisper num subprocesso.

    No Windows, uma DLL ausente (ex.: Visual C++ Redistributable) faz o
    ctranslate2 dar *segfault*, que derruba o processo sem exceção — o
    ``try/except`` não pega. Rodar o teste num processo separado permite cair
    para o modo sidecar com segurança.
    """
    import subprocess
    import sys

    code = ("from faster_whisper import WhisperModel; "
            f"WhisperModel({model_size!r}, device='cpu', compute_type='int8')")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=timeout_s)
    except (subprocess.TimeoutExpired, OSError):
        return False
    if r.returncode != 0:
        log.warning("Whisper falhou no teste (código %s): %s", r.returncode,
                    r.stderr.decode(errors="replace").strip().splitlines()[-1:] or "sem mensagem")
    return r.returncode == 0


class SidecarTranscriber:
    name = "sidecar"

    def transcribe(self, audio_path: str | Path) -> Transcript:
        p = Path(audio_path)
        js, txt = p.with_suffix(".json"), p.with_suffix(".txt")
        if js.exists():
            d = json.loads(_read_text(js))
            return Transcript(d["text"], d.get("language", "pt"), d.get("segments", []), "sidecar")
        if txt.exists():
            return Transcript(_read_text(txt).strip(), backend="sidecar")
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
    """``auto``: usa Whisper se ele passar no teste em subprocesso; senão, sidecar.

    A variável de ambiente ``HM_TRANSCRIBER`` (auto | whisper | sidecar) tem prioridade.
    """
    import os

    backend = os.getenv("HM_TRANSCRIBER", backend)
    if backend == "auto":
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return SidecarTranscriber()
        if not whisper_works(model_size):
            log.warning("Whisper indisponível neste ambiente; usando transcrição sidecar (.txt ao lado do áudio).")
            return SidecarTranscriber()
    if backend in ("auto", "whisper"):
        try:
            return WhisperTranscriber(model_size)
        except Exception as exc:  # sem pacote ou sem acesso ao modelo
            if backend == "whisper":
                raise
            log.warning("Whisper indisponível (%s); usando transcrição sidecar.", exc)
    return SidecarTranscriber()
