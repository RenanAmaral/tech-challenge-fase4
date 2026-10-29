from datetime import timedelta

import pytest

from hospital_monitor.events import Event, Modality
from hospital_monitor.fusion.engine import RiskFusionEngine
from hospital_monitor.orchestrator import Orchestrator


def ev(t, mod, score, kind="x", pid="P1"):
    return Event(patient_id=pid, modality=mod, kind=kind, anomaly_score=score, timestamp=t, description=f"{mod} {kind}")


def test_event_roundtrip(t0):
    e = ev(t0, Modality.AUDIO, 0.7, "voz_fadiga")
    back = Event.from_dict(e.to_dict())
    assert back.to_dict() == e.to_dict() and back.severity.value == "attention"


def test_noisy_or_and_agreement(t0):
    f = RiskFusionEngine()
    a = f.ingest(ev(t0, Modality.VITALS, 0.6))
    assert a.risk == pytest.approx(0.6)
    b = f.ingest(ev(t0 + timedelta(minutes=1), Modality.TEXT, 0.6))
    expected = 1 - (1 - 0.6) * (1 - 0.8 * 0.6) + 0.15
    assert b.risk == pytest.approx(expected) and b.level.value == "critical"
    assert b.agreeing == ["text", "vitals"]


def test_window_expiry(t0):
    f = RiskFusionEngine()
    f.ingest(ev(t0, Modality.VITALS, 0.9))
    later = f.ingest(ev(t0 + timedelta(minutes=20), Modality.AUDIO, 0.3))
    assert "vitals" not in later.contributions


def test_sentinel_not_attenuated(t0):
    a = RiskFusionEngine().ingest(ev(t0, Modality.VIDEO, 0.95, kind="queda"))
    assert a.level.value == "critical"


def test_alert_dedupe_update_escalation_ack(t0):
    o = Orchestrator()
    o.publish(ev(t0, Modality.VITALS, 0.6))                          # cria (atenção)
    o.publish(ev(t0 + timedelta(minutes=2), Modality.VITALS, 0.6))   # duplicado → suprimido
    assert len(o.alerts.alerts) == 1 and o.alerts.suppressed == 1
    o.publish(ev(t0 + timedelta(minutes=3), Modality.VITALS, 0.95))  # escalona (crítico)
    crit = [a for a in o.alerts.list() if a.level == "critical"]
    assert len(crit) == 1 and o.alerts.list()[0].status == "escalated"
    o.publish(ev(t0 + timedelta(minutes=4), Modality.AUDIO, 0.6))    # nova modalidade → atualização
    assert o.alerts.updates == 1 and "audio" in crit[0].modalities
    o.alerts.acknowledge(crit[0].alert_id, "Dra. Teste")
    assert crit[0].status == "acknowledged"


def test_publish_many_single_assessment(t0):
    o = Orchestrator()
    sent = []
    o.alerts.notifiers = [type("N", (), {"send": lambda self, a: sent.append(a)})()]
    o.publish_many([ev(t0, Modality.AUDIO, 0.9), ev(t0, Modality.TEXT, 1.0)])
    assert len(sent) == 1 and set(sent[0].modalities) == {"audio", "text"}


def test_api_flow(tmp_path):
    from fastapi.testclient import TestClient

    from hospital_monitor.api.app import create_app

    c = TestClient(create_app(tmp_path))
    assert c.get("/health").json()["status"] == "ok"
    base = {"patient_id": "API1", "hr": 78, "spo2": 97, "sbp": 120, "rr": 15, "temp": 36.7}
    from datetime import datetime, timezone

    t = datetime(2026, 10, 28, 8, 0, tzinfo=timezone.utc)
    for i in range(40):
        c.post("/vitals", json=base | {"timestamp": (t + timedelta(minutes=i)).isoformat()})
    alerts = []
    for i in range(40, 46):
        r = c.post("/vitals", json=base | {"spo2": 85, "hr": 112, "rr": 26,
                                           "timestamp": (t + timedelta(minutes=i)).isoformat()}).json()
        alerts += r["new_alerts"]
    assert alerts and alerts[-1]["level"] == "critical"
    aid = alerts[-1]["alert_id"]
    assert c.post(f"/alerts/{aid}/ack", json={"user": "Enf. API"}).json()["status"] == "acknowledged"
    assert c.get("/patients/API1/risk").json()["risk"] >= 0.8
