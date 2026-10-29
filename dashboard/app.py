"""Painel da equipe médica (Streamlit).

Executar:  streamlit run dashboard/app.py

Lê as saídas da demo em ``outputs/`` (ou de ``HM_OUTPUTS``). Se ``HM_API_URL``
estiver definido (ex.: http://localhost:8000), lê alertas da API e envia o
reconhecimento por ``POST /alerts/{id}/ack``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.getenv("HM_OUTPUTS", ROOT / "outputs"))
API = os.getenv("HM_API_URL")
LEVEL = {"critical": "🔴 crítico", "attention": "🟠 atenção", "info": "⚪ info"}
STATUS = {"open": "aberto", "acknowledged": "reconhecido", "escalated": "escalonado"}

st.set_page_config(page_title="Monitoramento multimodal", page_icon="🩺", layout="wide")


def load_alerts() -> list[dict]:
    if API:
        import requests

        return requests.get(f"{API}/alerts", timeout=5).json()
    p = OUT / "alerts.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def save_ack(alert_id: str, user: str) -> None:
    if API:
        import requests

        requests.post(f"{API}/alerts/{alert_id}/ack", json={"user": user}, timeout=5).raise_for_status()
        return
    p = OUT / "alerts.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    for a in data:
        if a["alert_id"] == alert_id:
            a["status"] = "acknowledged"
            a["acknowledged_by"] = user
            a["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_jsonl(name: str) -> pd.DataFrame:
    p = OUT / name
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()])


st.title("🩺 Monitoramento multimodal de pacientes")
st.caption(f"Fonte: {'API ' + API if API else OUT}")

alerts = load_alerts()
if not alerts:
    st.warning("Nenhum alerta encontrado. Rode `python scripts/run_demo.py` primeiro.")
    st.stop()

df = pd.DataFrame(alerts)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Alertas abertos", int((df.status == "open").sum()))
c2.metric("Críticos abertos", int(((df.status == "open") & (df.level == "critical")).sum()))
c3.metric("Reconhecidos", int((df.status == "acknowledged").sum()))
c4.metric("Pacientes com alerta", df.patient_id.nunique())

tab_alerts, tab_patient, tab_reports = st.tabs(["Alertas", "Paciente", "Relatórios"])

with tab_alerts:
    only_open = st.toggle("Somente abertos", value=True)
    user = st.text_input("Profissional que reconhece", value="Enf. plantão")
    view = df[df.status == "open"] if only_open else df
    for a in sorted(view.to_dict("records"), key=lambda a: (a["level"] != "critical", a["created_at"])):
        when = datetime.fromisoformat(a["created_at"]).strftime("%d/%m %H:%M")
        with st.container(border=True):
            left, right = st.columns([5, 1])
            left.markdown(f"**{LEVEL[a['level']]} · {a['patient_id']} · {when}** — risco {a['risk']:.2f}  \n{a['title']}")
            for d in a["details"][:5]:
                left.markdown(f"- {d}")
            left.caption(f"Modalidades: {', '.join(a['modalities'])} · status: {STATUS.get(a['status'], a['status'])}"
                         + (f" por {a['acknowledged_by']}" if a.get("acknowledged_by") else ""))
            if a["status"] == "open" and right.button("Reconhecer", key=a["alert_id"]):
                save_ack(a["alert_id"], user)
                st.rerun()

with tab_patient:
    ev = load_jsonl("events.jsonl")
    risk = load_jsonl("assessments.jsonl")
    pids = sorted(set(df.patient_id) | (set(ev.patient_id) if len(ev) else set()))
    pid = st.selectbox("Paciente", pids)
    if len(risk):
        r = risk[risk.patient_id == pid].copy()
        r["timestamp"] = pd.to_datetime(r["timestamp"], format="ISO8601")
        st.subheader("Risco fundido")
        st.line_chart(r.set_index("timestamp")["risk"], height=180)
    vit = OUT / "data" / f"vitais_{pid}.csv"
    if vit.exists():
        v = pd.read_csv(vit)
        v["timestamp"] = pd.to_datetime(v["timestamp"], format="ISO8601")
        st.subheader("Sinais vitais")
        st.line_chart(v.set_index("timestamp")[["spo2", "hr", "rr"]], height=240)
        st.caption(f"{int(v['flag'].sum())} minutos sinalizados pelo detector em streaming.")
    if len(ev):
        st.subheader("Eventos de anomalia")
        e = ev[ev.patient_id == pid][["timestamp", "modality", "kind", "anomaly_score", "description"]]
        st.dataframe(e, hide_index=True)

with tab_reports:
    reports = sorted((OUT / "video").glob("*.html")) + [OUT / "relatorio_demo.html"]
    choice = st.selectbox("Relatório", [p.name for p in reports if p.exists()])
    if choice:
        path = next(p for p in reports if p.name == choice)
        html = path.read_text(encoding="utf-8")
        if hasattr(st, "iframe"):  # Streamlit ≥ 1.5x
            st.iframe(html, height=900)
        else:  # versões antigas
            st.components.v1.html(html, height=900, scrolling=True)
    for mp4 in sorted((OUT / "video").glob("*.mp4")):
        st.caption(mp4.name)
        st.video(str(mp4))
