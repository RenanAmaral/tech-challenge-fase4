"""Relatório HTML consolidado da demo (outputs/relatorio_demo.html)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hospital_monitor.alerts.manager import MODALITY_PT
from hospital_monitor.anomaly.prescriptions import RULE_TEXT
from hospital_monitor.reporting import PALETTE, SEV_COLOR, esc, fig_to_b64, img, kind_pt, page, pill, style_axes, table

TZ = ZoneInfo("America/Sao_Paulo")


FIG_DIR: Path | None = None


def _save(name: str, b64: str) -> str:
    """Grava a figura em outputs/figuras (para README e relatório técnico) e devolve o base64."""
    import base64

    if FIG_DIR:
        (FIG_DIR / f"{name}.png").write_bytes(base64.b64decode(b64))
    return b64


def _hhmm(ts) -> str:
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return ts.astimezone(TZ).strftime("%d/%m %H:%M")


def _risk_chart(orch, patients) -> str:
    pids = list(patients)
    fig, axes = plt.subplots(len(pids), 1, figsize=(9, 1.45 * len(pids)), sharex=True)
    day0 = min(a.timestamp for a in orch.fusion.history).astimezone(TZ).replace(hour=0, minute=0)
    for ax, pid in zip(axes, pids):
        hist = [a for a in orch.fusion.history if a.patient_id == pid]
        # risco decai a zero quando a janela de 15 min expira sem novos eventos
        xs, ys = [day0], [0.0]
        for a in hist:
            t = a.timestamp.astimezone(TZ)
            if xs and (t - xs[-1]).total_seconds() > 15 * 60:
                xs += [xs[-1] + pd.Timedelta(minutes=15), xs[-1] + pd.Timedelta(minutes=15)]
                ys += [ys[-1], 0.0]
            xs += [t, t]
            ys += [ys[-1], a.risk]
        xs += [xs[-1] + pd.Timedelta(minutes=15), xs[-1] + pd.Timedelta(minutes=15), day0 + pd.Timedelta(hours=24)]
        ys += [ys[-1], 0.0, 0.0]
        ax.fill_between(xs, ys, color=PALETTE["line"], alpha=0.15, lw=0)
        ax.plot(xs, ys, color=PALETTE["line"], lw=1.2)
        for thr, col in ((0.5, PALETTE["warn"]), (0.8, PALETTE["crit"])):
            ax.axhline(thr, color=col, lw=0.8, ls=(0, (3, 3)))
        for al in orch.alerts.list(patient_id=pid):
            t = al.created_at.astimezone(TZ)
            ax.plot([t], [al.risk], marker="v", ms=7, color=SEV_COLOR[al.level], zorder=3)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 0.8])
        ax.text(0.003, 0.88, f"{pid} · {patients[pid]}", transform=ax.transAxes, fontsize=9, color=PALETTE["ink"])
        style_axes(ax)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=TZ))
    axes[-1].set_xlim(day0, day0 + pd.Timedelta(hours=24))
    axes[0].set_title("Risco fundido por paciente ao longo do dia (▼ = alerta emitido; tracejado = limiares 0,5 / 0,8)",
                      fontsize=10, loc="left")
    fig.tight_layout()
    return fig_to_b64(fig)


def _vitals_chart(df: pd.DataFrame, pid: str) -> str:
    t = pd.to_datetime(df["timestamp"]).dt.tz_convert(TZ)
    fig, axes = plt.subplots(3, 1, figsize=(9, 5), sharex=True)
    flag = df["flag"].to_numpy(bool)
    for ax, col, lab, thr in ((axes[0], "spo2", "SpO2 (%)", 92), (axes[1], "hr", "FC (bpm)", 120),
                              (axes[2], "rr", "FR (irpm)", 24)):
        ax.plot(t, df[col], color=PALETTE["line"], lw=1)
        ax.axhline(thr, color=PALETTE["warn"], lw=0.8, ls=(0, (3, 3)))
        ymin, ymax = ax.get_ylim()
        ax.fill_between(t, ymin, ymax, where=flag, color=PALETTE["band"], lw=0, step="mid")
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(lab, fontsize=9)
        style_axes(ax)
    axes[0].set_title(f"{pid} — sinais vitais (1/min). Faixa vermelha = episódio detectado em streaming; "
                      "tracejado = limiar clínico", fontsize=10, loc="left")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
    fig.tight_layout()
    return fig_to_b64(fig)


def _movement_chart(frames: dict) -> str:
    pids = [p for p in frames if frames[p]["pred"].any()]
    fig, axes = plt.subplots(len(pids), 1, figsize=(9, 1.9 * len(pids)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, pid in zip(axes, pids):
        df = frames[pid]
        t = pd.to_datetime(df["timestamp"]).dt.tz_convert(TZ)
        ax.bar(t, df["activity"], width=1 / 288, color=np.where(df["pred"], PALETTE["crit"], "#9FB3C8"))
        kinds = ", ".join(sorted({kind_pt(k) for k in df["flag"] if k}))
        ax.set_title(f"{pid} — atividade no leito (5 min); vermelho = {kinds}", fontsize=9.5, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh", tz=TZ))
        style_axes(ax)
    fig.tight_layout()
    return fig_to_b64(fig)


def _kpis(items) -> str:
    return "<div class='kpis'>" + "".join(f"<div class='kpi'><b>{v}</b><span>{esc(k)}</span></div>" for k, v in items) + "</div>"


def _metrics_section(m: dict) -> str:
    if not m:
        return "<p>Avaliação não executada (--skip-eval).</p>"
    v = m["sinais_vitais"]
    rows = [[esc(r["detector"]), r["precision_pontual"], r["recall_pontual"], r["f1_pontual"], r["precision_episodio"],
             r["recall_episodio"], r["atraso_medio_min"], r["falsos_episodios_por_dia"]] for r in v["detectores"]]
    html = (f"<h3>Sinais vitais — {v['n_series']} séries de 24 h, 5 tipos de anomalia cada</h3>"
            + table(["Detector", "Precisão", "Recall", "F1", "Precisão (episódio)", "Recall (episódio)",
                     "Atraso médio (min)", "Falsos episódios/dia"], rows)
            + f"<p class='meta'>Recall por tipo (combinado): {esc(v['recall_por_tipo'])}. "
              f"Falsos episódios em séries sem anomalia: {v['falsos_episodios_em_series_normais_por_dia']}/dia.</p>")
    p = m["prescricoes"]
    html += (f"<h3>Prescrições — {p['n_prescricoes']} prescrições, {p['n_anomalas']} anômalas</h3>"
             + table(["Método", "Precisão", "Recall", "F1"],
                     [["Regras farmacológicas", p["regras"]["precision"], p["regras"]["recall"], p["regras"]["f1"]],
                      ["Isolation Forest isolado", p["isolation_forest_isolado"]["precision"],
                       p["isolation_forest_isolado"]["recall"], p["isolation_forest_isolado"]["f1"]]]))
    mv = m["movimentacao"]
    html += (f"<h3>Movimentação — {mv['n_series']} séries de 48 h</h3>"
             + table(["Precisão (episódio)", "Recall (episódio)", "Atraso médio (min)", "Falsos alarmes em 10 séries normais"],
                     [[mv["precision_episodio"], mv["recall_episodio"], mv["atraso_medio_min"], mv["falsos_alarmes_em_48h_normais"]]]))
    ex = m["video_fisioterapia"]
    html += ("<h3>Vídeo — fisioterapia (keypoints COCO-17)</h3>"
             + table(["Desvio", "Recall por repetição"], [[esc(kind_pt(k)), v] for k, v in ex["recall_por_desvio"].items()])
             + f"<p class='meta'>Repetições normais marcadas indevidamente: {ex['reps_normais_marcadas']}. "
               f"Quedas detectadas: {ex['quedas_detectadas']}; quedas falsas: {ex['quedas_falsas_em_sessoes_normais']}.</p>")
    sx = m["video_cirurgia"]
    html += ("<h3>Vídeo — cirurgia (regras sobre detecções YOLOv8)</h3>"
             + table(["Falha", "Detectada"], [[esc(kind_pt(k)), v] for k, v in sx["detectados"].items()])
             + f"<p class='meta'>Achados em vídeos normais: {sx['achados_em_videos_normais']}.</p>")
    nl = m["nlp_clinico"]
    html += (f"<h3>NLP clínico — {nl['n_frases']} frases com negação</h3>"
             f"<p>Termos encontrados {nl['termos_encontrados']}; negação correta {nl['negacao_correta']} "
             f"(acurácia {nl['acuracia']}).</p>")
    if "voz_classificador" in m:
        vc = m["voz_classificador"]
        cm = vc["matrix"]
        html += (f"<h3>Voz — classificador Random Forest (validação cruzada 5-fold, n={vc['n']})</h3>"
                 f"<p>Acurácia: {vc['cv_accuracy']}. Features mais importantes: "
                 f"{esc(', '.join(list(vc['importances'])[:5]))}.</p>"
                 + table(["real \\ previsto", *vc["labels"]], [[lab, *row] for lab, row in zip(vc["labels"], cm)]))
    html += ("<p class='note'>Os dados são sintéticos e gerados com as mesmas famílias de anomalia que os detectores "
             "procuram; os números validam a implementação e a integração, não o desempenho clínico. A validação "
             "com dados reais (PhysioNet, TORGO, vídeos rotulados) está descrita no relatório técnico.</p>")
    return html


def build_demo_report(out: Path, orch, vit_frames, rx_frames, rx_findings, mv_frames, ex_results, sx_result,
                      au_results, metrics, patients) -> Path:
    global FIG_DIR
    FIG_DIR = out / "figuras"
    FIG_DIR.mkdir(exist_ok=True)
    alerts = orch.alerts.list()
    events = orch.bus.history
    kp = _kpis([
        ("pacientes monitorados", len(patients)),
        ("eventos de anomalia", len(events)),
        ("alertas críticos", sum(a.level == "critical" for a in alerts)),
        ("alertas de atenção", sum(a.level == "attention" for a in alerts)),
        ("atualizações de alerta", orch.alerts.updates),
        ("duplicados suprimidos", orch.alerts.suppressed),
    ])
    al_rows = []
    for a in alerts:
        st = "reconhecido por " + esc(a.acknowledged_by) + " às " + _hhmm(a.acknowledged_at) if a.acknowledged_by else "aberto"
        al_rows.append([_hhmm(a.created_at), esc(a.patient_id), pill(a.level), f"{a.risk:.2f}",
                        esc(", ".join(MODALITY_PT.get(m, m) for m in a.modalities)),
                        f"<b>{esc(a.title)}</b><br>" + "<br>".join("• " + esc(d) for d in a.details[:4]), st])
    ev_rows = [[_hhmm(e.timestamp), esc(e.patient_id), esc(MODALITY_PT.get(e.modality.value, e.modality.value)),
                esc(kind_pt(e.kind)), f"{e.anomaly_score:.2f}", esc(e.description)] for e in sorted(events, key=lambda e: e.timestamp)]

    au_rows = []
    for pid, r in au_results.items():
        v = r.voice
        au_rows.append([esc(pid), esc(r.transcript.text[:150]) + "…",
                        esc(", ".join(r.text.critical_terms) or "—"),
                        f"{r.text.sentiment['label']} ({r.text.sentiment['compound']:+.2f})",
                        "<br>".join(f"{k}: {s:.2f}" for k, s in v.scores.items()),
                        esc(v.main_condition or "normal"),
                        esc(", ".join(f"{k} {p:.2f}" for k, p in (v.classifier_probs or {}).items()))])
    feat_rows = [[esc(pid), *[f"{r.features.to_dict()[k]:.2f}" for k in
                              ("jitter_pct", "shimmer_pct", "hnr_db", "syllable_rate", "mean_pause_s",
                               "syllables_per_phrase", "breath_per_min")]] for pid, r in au_results.items()]

    ex_rows = []
    for pid, r in ex_results.items():
        ex_rows.append([esc(pid), len(r.reps), "<br>".join(esc(e.description) for e in r.events) or "—",
                        f"<a href='video/relatorio_fisioterapia_{pid}.html'>relatório</a> · "
                        f"<a href='video/fisioterapia_{pid}.mp4'>vídeo anotado</a>"])
    sx_rows = [[pill("critical" if f.score >= 0.8 else "attention"), esc(kind_pt(f.kind)), f"{f.start_s / 60:.1f} min",
                esc(f.description)] for f in sx_result.findings]
    rx_rows = [[esc(pid), esc(f.drug), esc(", ".join(RULE_TEXT.get(k, k) for k in f.kinds)), f"{f.score:.2f}",
                esc(f.description)] for pid, f in rx_findings]

    body = (
        kp
        + "<h2>1. Fluxo final: alertas à equipe médica</h2>"
        + "<p>Cada alerta nasce da fusão dos eventos de todas as modalidades na janela de 15 min do paciente. "
          "Alertas repetidos dentro de 30 min são anexados ao alerta aberto; escalonamentos sempre notificam.</p>"
        + table(["Quando", "Paciente", "Nível", "Risco", "Modalidades", "Conteúdo", "Status"], al_rows)
        + img(_save("risco_por_paciente", _risk_chart(orch, patients)), "risco")
        + "<h2>2. Sinais vitais em tempo real</h2>"
        + img(_save("vitais_P001", _vitals_chart(vit_frames["P001"], "P001")), "vitais P001")
        + img(_save("vitais_P003", _vitals_chart(vit_frames["P003"], "P003")), "vitais P003")
        + "<h2>3. Áudio das consultas</h2>"
        + table(["Paciente", "Transcrição", "Termos críticos", "Sentimento", "Escores de voz", "Achado de voz",
                 "Classificador RF"], au_rows)
        + "<h3>Features acústicas (Praat)</h3>"
        + table(["Paciente", "Jitter %", "Shimmer %", "HNR dB", "Sílabas/s", "Pausa média s", "Sílabas/frase",
                 "Inspirações/min"], feat_rows)
        + "<h2>4. Vídeo</h2><h3>Fisioterapia — pose + DTW</h3>"
        + table(["Paciente", "Repetições", "Achados", "Arquivos"], ex_rows)
        + "<h3>Cirurgia — YOLOv8 + regras de segurança</h3>"
        + table(["Nível", "Tipo", "Momento", "Descrição"], sx_rows)
        + "<p><a href='video/relatorio_cirurgia_P004.html'>Relatório completo do procedimento</a></p>"
        + "<h2>5. Prescrições</h2>" + table(["Paciente", "Fármaco", "Regra", "Escore", "Descrição"], rx_rows)
        + "<h2>6. Movimentação no leito</h2>" + img(_save("movimentacao", _movement_chart(mv_frames)), "movimentação")
        + "<h2>7. Todos os eventos de anomalia</h2>"
        + table(["Quando", "Paciente", "Modalidade", "Tipo", "Escore", "Descrição"], ev_rows)
        + "<h2>8. Avaliação quantitativa</h2>" + _metrics_section(metrics)
    )
    path = out / "relatorio_demo.html"
    path.write_text(page("Monitoramento multimodal — demo de 28/10/2026",
                         "Tech Challenge Fase 4 · horários de Brasília · dados 100% sintéticos", body), encoding="utf-8")
    return path
