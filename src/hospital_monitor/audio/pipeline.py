"""Pipeline de áudio: transcrição → features acústicas → voz + NLP → eventos."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ..events import Event, Modality
from .acoustic import AcousticFeatures, extract_features, load_audio
from .clinical_nlp import TextAnalysis, analyze_text
from .transcribe import Transcriber, Transcript
from .voice_analyzer import CONDITION_LABELS, VoiceAnalyzer, VoiceAssessment


@dataclass
class AudioResult:
    patient_id: str
    audio_path: str
    transcript: Transcript
    features: AcousticFeatures
    voice: VoiceAssessment
    text: TextAnalysis
    events: list[Event]

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "audio_path": self.audio_path,
            "transcript": {"text": self.transcript.text, "backend": self.transcript.backend,
                           "segments": self.transcript.segments},
            "acoustic_features": self.features.to_dict(),
            "voice_assessment": self.voice.to_dict(),
            "text_analysis": self.text.to_dict(),
            "events": [e.to_dict() for e in self.events],
        }


class AudioPipeline:
    def __init__(self, transcriber: Transcriber, voice_analyzer: VoiceAnalyzer, sr: int = 16000):
        self.transcriber = transcriber
        self.voice = voice_analyzer
        self.sr = sr

    def analyze(self, audio_path: str | Path, patient_id: str, timestamp: datetime,
                y: np.ndarray | None = None) -> AudioResult:
        y = load_audio(audio_path, self.sr) if y is None else y
        transcript = self.transcriber.transcribe(audio_path)
        feats = extract_features(y, self.sr)
        voice = self.voice.assess(feats)
        text = analyze_text(transcript.text)
        events: list[Event] = []
        if voice.main_condition:
            events.append(Event(
                patient_id=patient_id, modality=Modality.AUDIO, kind=f"voz_{voice.main_condition}",
                anomaly_score=voice.anomaly_score, timestamp=timestamp, source="audio.voice_analyzer",
                description=f"Voz com {CONDITION_LABELS[voice.main_condition]}.",
                evidence={"scores": voice.to_dict()["scores"], "drivers": voice.drivers,
                          "audio": str(audio_path), "classifier_probs": voice.classifier_probs},
            ))
        if text.anomaly_score >= 0.5:
            affirmed = [f.term for f in text.findings if f.effective_severity > 0]
            events.append(Event(
                patient_id=patient_id, modality=Modality.TEXT, kind="relato_sintomas",
                anomaly_score=text.anomaly_score, timestamp=timestamp, source="audio.clinical_nlp",
                description="Paciente relata: " + ", ".join(dict.fromkeys(affirmed)) + ".",
                evidence={"critical_terms": text.critical_terms, "sentiment": text.sentiment,
                          "transcript_excerpt": transcript.text[:240]},
            ))
        return AudioResult(patient_id, str(audio_path), transcript, feats, voice, text, events)
