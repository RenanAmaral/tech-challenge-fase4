"""Utilitários para relatórios HTML autocontidos (imagens embutidas em base64)."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {"ok": "#3B7D4F", "warn": "#C77C02", "crit": "#C0392B", "ink": "#1F2933", "muted": "#6B7785",
           "line": "#2F6FB0", "band": "#F3D9D5"}
SEV_COLOR = {"info": PALETTE["muted"], "attention": PALETTE["warn"], "critical": PALETTE["crit"]}
SEV_PT = {"info": "informativo", "attention": "atenção", "critical": "crítico"}

CSS = """
:root{--ink:#1F2933;--muted:#6B7785;--line:#E4E7EB;--bg:#FFFFFF;--soft:#F7F8FA;--crit:#C0392B;--warn:#C77C02;--ok:#3B7D4F}
*{box-sizing:border-box}body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
background:var(--bg);margin:0;line-height:1.5}main{max-width:1000px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:20px 0 6px}.meta{color:var(--muted);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{text-align:left;padding:6px 8px;
border-bottom:1px solid var(--line);vertical-align:top}th{background:var(--soft);font-weight:600}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600;color:#fff}
.kpi b .pill{font-size:13px;color:#fff;vertical-align:middle}.critical{background:var(--crit)}.attention{background:var(--warn)}.info{background:var(--muted)}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}.kpi{flex:1 1 150px;border:1px solid var(--line);
border-radius:8px;padding:10px 12px}.kpi b{display:block;font-size:22px}.kpi span{color:var(--muted);font-size:12px}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px}.grid{display:flex;flex-wrap:wrap;gap:12px}
.grid figure{margin:0;flex:1 1 260px}.grid figcaption{font-size:12px;color:var(--muted)}
pre{background:var(--soft);padding:10px;border-radius:6px;overflow:auto;font-size:12px}
.note{background:var(--soft);border-left:3px solid var(--muted);padding:8px 12px;font-size:13px}
"""


KIND_PT = {
    # vídeo
    "shallow": "amplitude reduzida", "asymmetric": "assimetria", "valgus": "valgo dinâmico",
    "trunk_lean": "inclinação do tronco", "atypical": "padrão atípico", "fall": "queda", "queda": "queda",
    "instrument_outside_field": "instrumento fora do campo", "camera_occlusion": "campo sem visibilidade",
    "gauze_count_mismatch": "contagem de gazes divergente", "instrument_count_mismatch": "contagem de instrumentos divergente",
    # prescrições
    "decimal_error": "erro de dose (10×)", "above_max": "acima da dose máxima", "duplication": "duplicidade terapêutica",
    "abrupt_change": "mudança abrupta", "padrao_incomum": "padrão incomum",
    # movimentação
    "agitacao_noturna": "agitação noturna", "imobilidade": "imobilidade",
    # sinais vitais
    "dessaturacao": "dessaturação", "hipotensao": "hipotensão", "hipertensao": "hipertensão",
    "deterioracao_silenciosa": "deterioração combinada", "taquicardia": "taquicardia", "febre": "febre",
    # áudio/texto
    "voz_fadiga": "voz: fadiga", "voz_disartria": "voz: disartria", "voz_dispneia": "voz: dispneia",
    "relato_sintomas": "relato de sintomas",
}


def kind_pt(kind: str) -> str:
    if kind.startswith("exercicio_"):
        return "exercício: " + kind.removeprefix("exercicio_").replace("_", " ")
    return KIND_PT.get(kind, kind.replace("_", " "))


def fmt_dt(dt) -> str:
    """Data/hora com o fuso explícito (ex.: 28/10/2026 10:00 (UTC−03:00))."""
    off = dt.utcoffset()
    if off is None or off.total_seconds() == 0:
        return f"{dt:%d/%m/%Y %H:%M} UTC"
    h = int(off.total_seconds() // 3600)
    return f"{dt:%d/%m/%Y %H:%M} (UTC{'+' if h >= 0 else '−'}{abs(h):02d}:00)"


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def file_to_b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def img(b64: str, alt: str = "") -> str:
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{b64}">'


def pill(sev: str) -> str:
    return f'<span class="pill {sev}">{SEV_PT.get(sev, sev)}</span>'


def esc(x) -> str:
    return html.escape(str(x))


def table(headers: list[str], rows: list[list]) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def page(title: str, subtitle: str, body: str) -> str:
    return (f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title>'
            f"<style>{CSS}</style></head><body><main><h1>{esc(title)}</h1><div class='meta'>{subtitle}</div>"
            f"{body}</main></body></html>")


def style_axes(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#B0B7C0")
    ax.spines["bottom"].set_color("#B0B7C0")
    ax.tick_params(colors="#4A5560", labelsize=9)
    ax.grid(axis="y", color="#ECEFF2", linewidth=0.8)
    ax.set_axisbelow(True)
