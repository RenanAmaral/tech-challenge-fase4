
from hospital_monitor.anomaly.metrics import event_level, pointwise
from hospital_monitor.anomaly.movement import MovementAnalyzer
from hospital_monitor.anomaly.prescriptions import PrescriptionAnalyzer
from hospital_monitor.anomaly.vitals import StreamingVitalsDetector
from hospital_monitor.data import synthetic as syn


def test_metrics_basics():
    s = pointwise([0, 1, 1, 0], [0, 1, 0, 1])
    assert (s.tp, s.fp, s.fn) == (1, 1, 1)
    ev = event_level([0, 1, 1, 0, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0])
    assert ev["tp"] == 1 and ev["fn"] == 1 and ev["mean_delay_min"] == 1.0


def test_vitals_desaturation_detected_and_typed():
    df = syn.generate_vitals(anomalies=[syn.VitalAnomaly("desaturation", 600, 60)], seed=3)
    out, events = StreamingVitalsDetector().process_frame(df)
    assert event_level(out["label"], out["flag"])["recall"] == 1.0
    assert any(e.kind == "dessaturacao" and e.anomaly_score >= 0.8 for e in events)
    # primeiro evento sai durante o episódio (streaming), não depois
    first = min(e.timestamp for e in events)
    assert df["timestamp"][600] <= first <= df["timestamp"][630]


def test_vitals_silent_deterioration_caught_by_multivariate():
    df = syn.generate_vitals(anomalies=[syn.VitalAnomaly("silent_deterioration", 600, 150)], seed=4)
    out, events = StreamingVitalsDetector().process_frame(df)
    assert out.loc[out.label == 1, "threshold_score"].max() < 0.5  # nenhum limiar clínico cruzado
    assert out.loc[out.label == 1, "flag"].any()


def test_vitals_normal_series_has_few_false_alarms():
    out, _ = StreamingVitalsDetector().process_frame(syn.generate_vitals(seed=123))
    assert event_level(out["label"], out["flag"])["fp"] <= 1


def test_prescription_rules():
    an = PrescriptionAnalyzer()
    df = syn.generate_prescriptions(anomalies=[{"kind": k, "at_hours": 30 + 5 * i} for i, k in
                                               enumerate(["decimal_error", "above_max", "duplication", "abrupt_change"])])
    f, events, _ = an.analyze(df)
    assert pointwise(f["label"], f["flag"]).f1 == 1.0
    kinds = {e.kind for e in events}
    assert {"decimal_error", "above_max", "duplication", "abrupt_change"} <= kinds


def test_movement_patterns():
    df = syn.generate_movement(anomalies=[{"kind": "immobility", "start_h": 10, "duration_h": 3},
                                          {"kind": "night_agitation", "start_h": 26, "duration_h": 1.5},
                                          {"kind": "fall", "start_h": 40}], seed=1)
    out, events = MovementAnalyzer().analyze(df)
    assert {e.kind for e in events} == {"imobilidade", "agitacao_noturna", "queda"}
    _, normal = MovementAnalyzer().analyze(syn.generate_movement(seed=77))
    assert normal == []


def test_bidmc_loader(tmp_path):
    from hospital_monitor.data.physionet import load_bidmc_numerics

    p = tmp_path / "bidmc_01_Numerics.csv"
    rows = ["Time [s], HR, PULSE, RESP, SpO2"] + [f"{i},{80 + i % 3},{80},{18},{97}" for i in range(600)]
    p.write_text("\n".join(rows))
    df = load_bidmc_numerics(p)
    assert len(df) == 10 and set(df.attrs["imputed"]) == {"sbp", "dbp", "temp"}
    out, _ = StreamingVitalsDetector(warmup=3).process_frame(df)
    assert not out["flag"].any()
