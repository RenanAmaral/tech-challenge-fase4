"""Leitura de sinais vitais reais do PhysioNet.

Suporte a:
  * **BIDMC PPG and Respiration** (aberto, sem credenciamento):
    ``bidmc_XX_Numerics.csv`` com HR, PULSE, RESP, SpO2 a 1 Hz.
    https://physionet.org/content/bidmc/1.0.0/
  * CSV genérico com colunas ``timestamp`` + qualquer subconjunto de
    ``hr, spo2, sbp, dbp, rr, temp`` (ex.: exportação do MIMIC-IV Demo).

Sinais ausentes são preenchidos com valores fisiológicos constantes e
listados em ``df.attrs["imputed"]`` — o detector então só "enxerga" os sinais
medidos. Baixe os arquivos com ``python scripts/download_physionet.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DEFAULTS = {"hr": 75.0, "spo2": 97.0, "sbp": 120.0, "dbp": 78.0, "rr": 15.0, "temp": 36.8}
BIDMC_URL = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/bidmc_{:02d}_Numerics.csv"


def _finish(df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    imputed = []
    for k, v in DEFAULTS.items():
        if k not in df or df[k].isna().all():
            df[k] = v
            imputed.append(k)
        else:
            df[k] = df[k].interpolate(limit_direction="both")
    df["patient_id"] = patient_id
    df["label"] = 0
    df["anomaly_type"] = ""
    df.attrs["imputed"] = imputed
    return df[["timestamp", "patient_id", *DEFAULTS, "label", "anomaly_type"]]


def load_bidmc_numerics(path: str | Path, patient_id: str | None = None, start: datetime | None = None,
                        resample: str = "1min") -> pd.DataFrame:
    """Lê ``bidmc_XX_Numerics.csv`` e reamostra para 1 amostra/min (média)."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    start = start or datetime(2026, 10, 28, tzinfo=timezone.utc)
    t = pd.to_datetime([start + timedelta(seconds=float(s)) for s in raw["Time [s]"]])
    df = pd.DataFrame({"timestamp": t, "hr": raw.get("HR"), "spo2": raw.get("SpO2"), "rr": raw.get("RESP")})
    df = df.set_index("timestamp").resample(resample).mean().reset_index()
    return _finish(df, patient_id or Path(path).stem.split("_Numerics")[0])


def load_vitals_csv(path: str | Path, patient_id: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    pid = patient_id or (df["patient_id"].iloc[0] if "patient_id" in df else Path(path).stem)
    return _finish(df, str(pid))
