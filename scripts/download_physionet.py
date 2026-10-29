"""Baixa registros do BIDMC (PhysioNet, acesso aberto) e roda o detector de sinais vitais.

Uso:
    python scripts/download_physionet.py --records 1 2 3
    python scripts/download_physionet.py --records 1 --inject desaturation   # injeta anomalia p/ teste

Os arquivos ficam em data/physionet/. Cada registro tem ~8 min; com
``--inject`` uma dessaturação sintética é sobreposta para validar a detecção
sobre ruído fisiológico real.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hospital_monitor.anomaly.metrics import event_level  # noqa: E402
from hospital_monitor.anomaly.vitals import StreamingVitalsDetector  # noqa: E402
from hospital_monitor.data.physionet import BIDMC_URL, load_bidmc_numerics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--inject", choices=["desaturation"], default=None)
    args = ap.parse_args()
    out = ROOT / "data" / "physionet"
    out.mkdir(parents=True, exist_ok=True)
    for r in args.records:
        p = out / f"bidmc_{r:02d}_Numerics.csv"
        if not p.exists():
            print("baixando", BIDMC_URL.format(r))
            urllib.request.urlretrieve(BIDMC_URL.format(r), p)
        # BIDMC tem 1 Hz por ~8 min: usar 1 amostra/5 s para ter série suficiente
        df = load_bidmc_numerics(p, resample="5s")
        if args.inject:
            n = len(df)
            s, e = int(n * 0.6), int(n * 0.85)
            df.loc[s:e, "spo2"] -= 10
            df.loc[s:e, "label"] = 1
        det = StreamingVitalsDetector(warmup=20, baseline_samples=60)
        res, events = det.process_frame(df)
        print(f"{p.name}: {len(df)} amostras, sinais imputados={df.attrs['imputed']}, eventos={len(events)}")
        for e in events:
            print("   ", e.timestamp.time(), e.kind, round(e.anomaly_score, 2), e.description[:90])
        if args.inject:
            print("    detecção:", event_level(res["label"], res["flag"], step_minutes=5 / 60))


if __name__ == "__main__":
    main()
