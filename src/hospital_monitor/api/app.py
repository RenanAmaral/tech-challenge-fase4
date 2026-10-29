"""API REST (FastAPI) para ingestão e consulta.

Executar:  ``uvicorn hospital_monitor.api.app:app --reload``  (docs em /docs)

Endpoints principais:
  POST /vitals                 amostra de sinais vitais (streaming, 1 por chamada)
  POST /events                 evento já processado por outro serviço
  POST /analyze/audio          WAV da consulta (+ transcrição opcional)
  POST /analyze/exercise       keypoints JSON de sessão de fisioterapia
  POST /analyze/prescriptions  CSV de prescrições
  GET  /alerts                 alertas (filtro por status/paciente)
  POST /alerts/{id}/ack        reconhecimento do alerta pela equipe
  GET  /patients/{id}/risk     risco fundido atual
"""

from __future__ import annotations

import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..alerts.notifiers import FileNotifier, notifiers_from_env
from ..config import load_settings
from ..events import Event
from ..orchestrator import Orchestrator
from .. import services


class VitalSample(BaseModel):
    patient_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hr: float
    spo2: float
    sbp: float
    dbp: float | None = None
    rr: float
    temp: float = 36.8


class AckBody(BaseModel):
    user: str = "equipe"


def create_app(outputs_dir: str | Path | None = None) -> FastAPI:
    settings = load_settings()
    out = Path(outputs_dir) if outputs_dir else settings.outputs_dir / "api"
    orch = Orchestrator(settings, [FileNotifier(out / "alerts.jsonl"), *notifiers_from_env()], out)
    vitals = services.make_vitals_detector(settings)
    state: dict = {}

    app = FastAPI(title="Hospital Monitor — monitoramento multimodal", version="1.0.0")

    def _publish(events: list[Event]) -> list[dict]:
        before = set(orch.alerts.alerts)
        orch.publish_many(events)  # eventos da mesma análise → uma avaliação
        return [a.to_dict() for k, a in orch.alerts.alerts.items() if k not in before]

    @app.get("/health")
    def health():
        return {"status": "ok", "patients": sorted({e.patient_id for e in orch.bus.history}),
                "alerts_open": len(orch.alerts.list("open"))}

    @app.post("/vitals")
    def post_vital(sample: VitalSample):
        s = sample.model_dump()
        if s["timestamp"].tzinfo is None:
            s["timestamp"] = s["timestamp"].replace(tzinfo=timezone.utc)
        sc, events = vitals.update(sample.patient_id, s)
        return {"score": round(sc.score, 3), "reasons": sc.reasons, "events": [e.to_dict() for e in events],
                "new_alerts": _publish(events)}

    @app.post("/events")
    def post_event(event: dict):
        try:
            ev = Event.from_dict(event)
        except Exception as exc:
            raise HTTPException(422, f"evento inválido: {exc}") from exc
        return {"event": ev.to_dict(), "new_alerts": _publish([ev])}

    @app.post("/analyze/audio")
    async def analyze_audio(patient_id: str = Form(...), audio: UploadFile = File(...),
                            transcript: str | None = Form(None), timestamp: datetime | None = Form(None)):
        if "audio" not in state:
            state["audio"] = services.make_audio_pipeline("auto")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / (audio.filename or "consulta.wav")
            p.write_bytes(await audio.read())
            if transcript:
                p.with_suffix(".txt").write_text(transcript, encoding="utf-8")
            ts = timestamp or datetime.now(timezone.utc)
            res = state["audio"].analyze(p, patient_id, ts)
        d = res.to_dict()
        d["new_alerts"] = _publish(res.events)
        return d

    @app.post("/analyze/exercise")
    async def analyze_exercise(patient_id: str = Form(...), keypoints: UploadFile = File(...)):
        if "exercise" not in state:
            state["exercise"] = services.make_exercise_analyzer()
        d = json.loads(await keypoints.read())
        kp, fps = np.asarray(d["keypoints"], float), float(d.get("fps", 15))
        res = state["exercise"].analyze(kp, fps, patient_id, datetime.now(timezone.utc))
        s = res.summary()
        s["new_alerts"] = _publish(res.events)
        return s

    @app.post("/analyze/prescriptions")
    async def analyze_prescriptions(file: UploadFile = File(...)):
        if "rx" not in state:
            state["rx"] = services.make_prescription_analyzer()
        df = pd.read_csv(io.BytesIO(await file.read()), parse_dates=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        _, events, findings = state["rx"].analyze(df)
        return {"findings": [f.__dict__ for f in findings], "new_alerts": _publish(events)}

    @app.get("/alerts")
    def get_alerts(status: str | None = None, patient_id: str | None = None):
        return [a.to_dict() for a in orch.alerts.list(status, patient_id)]

    @app.post("/alerts/{alert_id}/ack")
    def ack(alert_id: str, body: AckBody):
        if alert_id not in orch.alerts.alerts:
            raise HTTPException(404, "alerta não encontrado")
        return orch.alerts.acknowledge(alert_id, body.user).to_dict()

    @app.get("/patients/{patient_id}/risk")
    def risk(patient_id: str):
        a = orch.fusion.current_risk(patient_id)
        if not a:
            raise HTTPException(404, "sem dados para o paciente")
        return a.to_dict()

    @app.get("/patients/{patient_id}/events")
    def patient_events(patient_id: str, limit: int = 50):
        evs = [e.to_dict() for e in orch.bus.history if e.patient_id == patient_id]
        return evs[-limit:]

    app.state.orchestrator = orch
    return app


app = create_app()
