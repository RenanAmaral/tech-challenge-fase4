"""Detecção de alterações inesperadas na evolução das prescrições.

Regras farmacológicas (explicáveis, alta precisão) + Isolation Forest sobre
features da alteração (detector estatístico de padrões incomuns).

Regras:
  * ``decimal_error`` — dose 8–12× (ou 1/8–1/12) da anterior do mesmo fármaco
    (erro de vírgula/zero, causa clássica de evento adverso);
  * ``above_max`` — dose diária acima do máximo de referência;
  * ``duplication`` — dois fármacos ativos da mesma classe de alto risco
    (ex.: dois anticoagulantes);
  * ``abrupt_change`` — dose diária ≥ 3× ou ≤ 1/3 da anterior.

A tabela de referência é simplificada e didática; em produção viria de uma
base farmacológica validada.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..events import Event, Modality

# fármaco → (classe, dose diária máxima, unidade)
DRUG_REFERENCE: dict[str, tuple[str, float, str]] = {
    "dipirona": ("analgesico", 4000, "mg"),
    "paracetamol": ("analgesico", 4000, "mg"),
    "morfina": ("opioide", 60, "mg"),
    "tramadol": ("opioide", 400, "mg"),
    "enoxaparina": ("anticoagulante", 200, "mg"),
    "heparina": ("anticoagulante", 40000, "UI"),
    "varfarina": ("anticoagulante", 15, "mg"),
    "losartana": ("anti-hipertensivo", 100, "mg"),
    "captopril": ("anti-hipertensivo", 150, "mg"),
    "metformina": ("antidiabetico", 2550, "mg"),
    "furosemida": ("diuretico", 600, "mg"),
    "omeprazol": ("ibp", 80, "mg"),
    "ceftriaxona": ("antibiotico", 4000, "mg"),
    "amoxicilina": ("antibiotico", 3000, "mg"),
}
HIGH_RISK_CLASSES = {"anticoagulante", "opioide"}

RULE_SCORES = {"decimal_error": 0.95, "above_max": 0.9, "duplication": 0.85, "abrupt_change": 0.7}
RULE_TEXT = {
    "decimal_error": "possível erro de dose (fator 10×)",
    "above_max": "dose diária acima do máximo recomendado",
    "duplication": "duplicidade terapêutica em classe de alto risco",
    "abrupt_change": "mudança abrupta de dose",
}
FEATURES = ["log_ratio", "daily_over_max", "changes_24h", "same_class_active", "hour"]


@dataclass
class PrescriptionFinding:
    index: int
    drug: str
    kinds: list[str]
    score: float
    description: str


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por prescrição com features relativas ao histórico do paciente."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    feats, active, history = [], {}, []
    for _, r in df.iterrows():
        cls, mx, _ = DRUG_REFERENCE.get(r["drug"], ("desconhecida", np.inf, r["unit"]))
        daily = r["dose"] * r["freq_per_day"]
        prev = active.get(r["drug"])
        dose_ratio = r["dose"] / prev["dose"] if prev is not None else 1.0
        daily_ratio = daily / (prev["dose"] * prev["freq_per_day"]) if prev is not None else 1.0
        same = sum(1 for d, a in active.items() if d != r["drug"] and a["class"] == cls)
        recent = sum(1 for t, d in history if d == r["drug"] and (r["timestamp"] - t).total_seconds() < 86400)
        feats.append({
            "dose_ratio": dose_ratio, "daily_ratio": daily_ratio, "log_ratio": float(np.log(daily_ratio)),
            "daily": daily, "daily_over_max": float(daily / mx) if np.isfinite(mx) else 0.0,
            "changes_24h": recent, "same_class_active": same, "class": cls,
            "hour": r["timestamp"].hour + r["timestamp"].minute / 60,
        })
        active[r["drug"]] = {"dose": r["dose"], "freq_per_day": r["freq_per_day"], "class": cls}
        history.append((r["timestamp"], r["drug"]))
    return pd.concat([df, pd.DataFrame(feats)], axis=1)


class PrescriptionAnalyzer:
    """Regras decidem o alerta; o Isolation Forest apenas anota padrões incomuns.

    Na avaliação com dados sintéticos o Isolation Forest não separou bem ajustes
    rotineiros de erros (ver relatório técnico), então seus achados isolados
    recebem escore 0,45 — visíveis no relatório, abaixo do limiar de alerta —
    para não gerar fadiga de alertas.
    """

    IF_INFO_SCORE = 0.45

    def __init__(self, iforest=None, if_threshold: float | None = None):
        self.iforest = iforest  # IsolationForest treinado em histórico normal (opcional)
        self.if_threshold = if_threshold

    @staticmethod
    def rules(row: pd.Series) -> list[str]:
        kinds = []
        if 8 <= row["dose_ratio"] <= 12 or 1 / 12 <= row["dose_ratio"] <= 1 / 8:
            kinds.append("decimal_error")
        if row["daily_over_max"] > 1.0:
            kinds.append("above_max")
        if row["class"] in HIGH_RISK_CLASSES and row["same_class_active"] > 0:
            kinds.append("duplication")
        if not kinds and (row["daily_ratio"] >= 3 or row["daily_ratio"] <= 1 / 3):
            kinds.append("abrupt_change")
        return kinds

    @classmethod
    def fit_iforest(cls, normal_feats: pd.DataFrame, seed: int = 0) -> tuple[object, float]:
        """Treina em histórico normal; limiar = maior escore visto no treino."""
        from sklearn.ensemble import IsolationForest

        X = normal_feats[FEATURES].abs().to_numpy()
        m = IsolationForest(n_estimators=200, contamination="auto", random_state=seed).fit(X)
        return m, float(np.max(-m.score_samples(X)))

    def analyze(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[Event], list[PrescriptionFinding]]:
        f = build_features(df)
        f["rule_kinds"] = f.apply(self.rules, axis=1)
        f["rule_score"] = f["rule_kinds"].apply(lambda ks: max((RULE_SCORES[k] for k in ks), default=0.0))
        if self.iforest is not None:
            raw = -self.iforest.score_samples(f[FEATURES].abs().to_numpy())  # maior = mais anômalo
            thr = self.if_threshold if self.if_threshold is not None else np.inf
            f["if_raw"] = raw
            f["if_flag"] = raw > thr
            f["if_score"] = np.where(f["if_flag"], self.IF_INFO_SCORE, 0.0)
        else:
            f["if_raw"], f["if_flag"], f["if_score"] = 0.0, False, 0.0
        f["score"] = f[["rule_score", "if_score"]].max(axis=1)
        f["flag"] = f["score"] >= 0.5
        events, findings = [], []
        for i, r in f[f["flag"]].iterrows():
            kinds = r["rule_kinds"] or ["padrao_incomum"]
            txt = "; ".join(RULE_TEXT.get(k, "alteração estatisticamente incomum") for k in kinds)
            desc = (f"Prescrição de {r['drug']} {r['dose']:g} {r['unit']} {r['freq_per_day']}x/dia "
                    f"({r['daily']:g} {r['unit']}/dia): {txt}.")
            findings.append(PrescriptionFinding(i, r["drug"], kinds, float(r["score"]), desc))
            events.append(Event(
                patient_id=r["patient_id"], modality=Modality.PRESCRIPTION, kind=kinds[0],
                anomaly_score=float(r["score"]), timestamp=r["timestamp"].to_pydatetime(),
                source="anomaly.prescriptions", description=desc,
                evidence={"drug": r["drug"], "dose": r["dose"], "freq_per_day": int(r["freq_per_day"]),
                          "daily": r["daily"], "daily_ratio": round(float(r["daily_ratio"]), 2),
                          "daily_over_max": round(float(r["daily_over_max"]), 2), "prescriber": r["prescriber"]},
            ))
        return f, events, findings
