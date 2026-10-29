"""Relatórios automáticos de vídeo (fisioterapia e cirurgia) em HTML."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..reporting import PALETTE, esc, fig_to_b64, fmt_dt, img, kind_pt, page, pill, style_axes, table
from .exercise_analyzer import DEVIATION_TEXT, ExerciseResult
from .render import LABEL, draw_frame
from .surgical import SurgicalResult


def _angles_chart(res: ExerciseResult) -> str:
    t = np.arange(res.n_frames) / res.fps
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for d in res.deviations:
        ax.axvspan(d.start_frame / res.fps, d.end_frame / res.fps, color=PALETTE["band"], lw=0)
    ax.plot(t, res.angles["knee_l"], color=PALETTE["line"], lw=1.6, label="Joelho esquerdo")
    ax.plot(t, res.angles["knee_r"], color="#8E5BB5", lw=1.6, label="Joelho direito")
    ax.plot(t, res.angles["trunk_lean"], color=PALETTE["muted"], lw=1.2, ls="--", label="Inclinação do tronco")
    ax.set_xlabel("tempo (s)", fontsize=9)
    ax.set_ylabel("ângulo (°)", fontsize=9)
    ax.set_title("Ângulos articulares — faixas em vermelho indicam desvios detectados", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    style_axes(ax)
    return fig_to_b64(fig)


def _frame_b64(kp, frame, label, bad) -> str:
    import base64

    import cv2

    ok, buf = cv2.imencode(".png", draw_frame(kp[frame], 300, bad, label))
    return base64.b64encode(buf.tobytes()).decode()


def exercise_report(res: ExerciseResult, kp: np.ndarray, path: str | Path, session_name: str,
                    started: datetime) -> Path:
    s = res.summary()
    worst = max((e.severity.value for e in res.events), key=lambda v: ["info", "attention", "critical"].index(v),
                default="info")
    kpis = (f"<div class='kpis'><div class='kpi'><b>{s['n_reps']}</b><span>repetições</span></div>"
            f"<div class='kpi'><b>{s['n_deviations']}</b><span>desvios</span></div>"
            f"<div class='kpi'><b>{s['duration_s']} s</b><span>duração</span></div>"
            f"<div class='kpi'><b>{pill(worst) if res.events else pill('info')}</b><span>nível</span></div></div>")
    rows = [[r["rep"] + 1, f"{r['rom']:.0f}°", f"{r['asym_max']:.0f}°", f"{r['valgus_max']:.1f} cm",
             f"{r['trunk_max']:.0f}°", f"{r['dtw']:.1f}",
             ", ".join(kind_pt(d["kind"]) for d in s["deviations"] if d["rep"] == r["rep"]) or "—"] for r in s["reps"]]
    dev_rows = [[esc(kind_pt(d.kind)), d.rep + 1 if d.rep is not None else "—", f"{d.start_frame / res.fps:.1f}–{d.end_frame / res.fps:.1f} s",
                 f"{d.measured:.1f}", esc(d.expected), f"{d.score:.2f}", esc(DEVIATION_TEXT[d.kind])] for d in res.deviations]
    figs = ""
    for d in res.deviations[:6]:
        mid = (d.start_frame + d.end_frame) // 2 if d.kind != "fall" else min(d.end_frame, len(kp) - 1)
        if d.kind == "shallow" or d.kind == "asymmetric":
            seg = res.angles["knee_l"][d.start_frame : d.end_frame + 1]
            mid = d.start_frame + int(np.argmin(seg))
        cap = f"Rep. {d.rep + 1 if d.rep is not None else '—'} · {kind_pt(d.kind)} · t={mid / res.fps:.1f}s"
        figs += (f"<figure>{img(_frame_b64(kp, mid, LABEL.get(d.kind, d.kind).upper(), True))}"
                 f"<figcaption>{esc(cap)}</figcaption></figure>")
    if res.reps:
        r0 = next((r for r in res.reps if not any(d.rep == r.rep for d in res.deviations)), res.reps[0])
        mid = r0.start_frame + int(np.argmin(res.angles["knee_l"][r0.start_frame : r0.end_frame + 1]))
        figs = (f"<figure>{img(_frame_b64(kp, mid, 'DENTRO DO PADRAO', False))}<figcaption>Execução dentro do padrão "
                f"(rep. {r0.rep + 1})</figcaption></figure>") + figs
    events = "".join(f"<li>{pill(e.severity.value)} {esc(e.description)}</li>" for e in res.events) or "<li>Nenhum desvio.</li>"
    body = (kpis + "<h2>Achados</h2><ul>" + events + "</ul>"
            + "<h2>Ângulos ao longo da sessão</h2>" + img(_angles_chart(res), "ângulos")
            + "<h2>Quadros-chave</h2><div class='grid'>" + figs + "</div>"
            + "<h2>Métricas por repetição</h2>"
            + table(["Rep.", "ROM joelho", "Assimetria E/D", "Valgo", "Tronco máx.", "DTW", "Desvios"], rows)
            + "<h2>Desvios detectados</h2>"
            + (table(["Tipo", "Rep.", "Intervalo", "Medido", "Esperado", "Escore", "Interpretação"], dev_rows)
               if dev_rows else "<p>Nenhum.</p>")
            + "<p class='note'>Relatório gerado automaticamente para apoio à equipe de fisioterapia. "
              "Não substitui a avaliação do profissional.</p>")
    sub = f"Paciente {esc(res.patient_id)} · {esc(session_name)} · {fmt_dt(started)}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page("Relatório de sessão de fisioterapia", sub, body), encoding="utf-8")
    return path


def surgical_report(res: SurgicalResult, path: str | Path, procedure: str, started: datetime) -> Path:
    s = res.summary()
    t = np.arange(res.n_frames) / res.fps / 60
    fig, ax = plt.subplots(figsize=(9, 3.0))
    for f in res.findings:
        if f.end_s > f.start_s:
            ax.axvspan(f.start_s / 60, f.end_s / 60, color=PALETTE["band"], lw=0)
    for c, col in (("gaze", PALETTE["crit"]), ("pinca", PALETTE["line"]), ("bisturi", "#8E5BB5")):
        if c in res.counts_series:
            x = np.convolve(res.counts_series[c], np.ones(15) / 15, mode="same")
            ax.plot(t, x, color=col, lw=1.5, label=c)
    ax.set_xlabel("tempo (min)", fontsize=9)
    ax.set_ylabel("objetos detectados", fontsize=9)
    ax.set_title("Detecções YOLOv8 por classe (média móvel 3 s) — faixas: eventos", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    style_axes(ax)
    rows = [[pill("critical" if f.score >= 0.8 else "attention"), esc(kind_pt(f.kind)), f"{f.start_s / 60:.1f} min",
             esc(f.description)] for f in res.findings]
    cnt = [[esc(c), res.counts_start.get(c, 0), res.counts_end.get(c, 0)] for c in sorted(res.counts_start) if c != "mao"]
    body = ("<h2>Achados</h2>" + (table(["Nível", "Tipo", "Momento", "Descrição"], rows) if rows else "<p>Nenhum.</p>")
            + "<h2>Contagem de materiais</h2>" + table(["Classe", "Início", "Fim"], cnt)
            + "<h2>Linha do tempo</h2>" + img(fig_to_b64(fig))
            + f"<p class='meta'>Duração analisada: {s['duration_s'] / 60:.1f} min a {res.fps:.0f} fps.</p>"
            + "<p class='note'>Relatório gerado automaticamente. A contagem de compressas deve ser confirmada "
              "pela equipe de enfermagem conforme protocolo de cirurgia segura (OMS).</p>")
    sub = f"Paciente {esc(res.patient_id)} · {esc(procedure)} · {fmt_dt(started)}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page("Relatório automático de procedimento cirúrgico", sub, body), encoding="utf-8")
    return path
