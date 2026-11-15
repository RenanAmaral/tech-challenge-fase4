import numpy as np
import pytest

from hospital_monitor.data import synthetic as syn
from hospital_monitor.video import kinematics as K
from hospital_monitor.video.exercise_analyzer import ExerciseAnalyzer, build_reference
from hospital_monitor.video.surgical import SurgicalVideoAnalyzer


@pytest.fixture(scope="module")
def analyzer():
    s = syn.generate_squat_session(reps=8, seed=100)
    return ExerciseAnalyzer(build_reference(s.keypoints, s.fps))


def test_angle_and_dtw():
    a, b, c = np.array([[1.0, 0, 0]]), np.zeros((1, 3)), np.array([[0, 1.0, 0]])
    assert K.angle(a, b, c)[0] == pytest.approx(90)
    x = np.sin(np.linspace(0, 6, 60))
    assert K.dtw_distance(x, x) == pytest.approx(0)
    assert K.dtw_distance(x, np.roll(x, 5)) < K.dtw_distance(x, -x)


def test_interpolate_missing():
    x = np.array([[0.0], [np.nan], [2.0]])
    assert K.interpolate_missing(x)[1, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("dev", ["shallow", "asymmetric", "valgus", "trunk_lean"])
def test_deviation_detected_on_right_reps(analyzer, dev, t0):
    s = syn.generate_squat_session(reps=6, deviation=dev, deviation_reps=(1, 4), seed=5)
    r = analyzer.analyze(s.keypoints, s.fps, "P", t0)
    assert len(r.reps) == 6
    assert {d.rep for d in r.deviations if d.kind == dev} == {1, 4}
    assert all(0.5 <= e.anomaly_score <= 0.75 for e in r.events)


def test_normal_session_clean(analyzer, t0):
    s = syn.generate_squat_session(reps=6, seed=9)
    assert analyzer.analyze(s.keypoints, s.fps, "P", t0).deviations == []


def test_reference_ignores_partial_reps():
    """Movimento parcial na gravação de referência não entra no padrão (nem afrouxa o DTW)."""
    clean = syn.generate_squat_session(reps=6, seed=100)
    noisy = syn.generate_squat_session(reps=6, deviation="shallow", deviation_reps=(5,), seed=100)
    ref_clean, ref_noisy = build_reference(clean.keypoints, clean.fps), build_reference(noisy.keypoints, noisy.fps)
    assert ref_noisy.n_reps == 5
    assert ref_noisy.dtw_std <= ref_clean.dtw_std * 1.5 + 0.5


def test_balance_loss_step(analyzer, t0):
    """Pisar em falso: um pé sai do lugar no meio da repetição (ângulos quase normais)."""
    s = syn.generate_squat_session(reps=6, seed=9)
    kp = s.keypoints.copy()
    reps = analyzer.analyze(kp, s.fps, "P", t0).reps
    st, en = reps[2].start_frame, reps[2].end_frame
    mid = (st + en) // 2
    kp[mid : en + 1, 16, 0] += 0.25  # tornozelo direito dá um passo de 25 cm para o lado
    r = analyzer.analyze(kp, s.fps, "P", t0)
    assert {d.rep for d in r.deviations if d.kind == "balance"} == {2}
    assert any(e.kind == "exercicio_perda_equilibrio" and 0.5 <= e.anomaly_score <= 0.75 for e in r.events)


def test_fall(analyzer, t0):
    s = syn.generate_squat_session(reps=5, fall_at_rep=2, seed=11)
    r = analyzer.analyze(s.keypoints, s.fps, "P", t0)
    assert any(e.kind == "queda" and e.anomaly_score >= 0.9 for e in r.events)
    # o intervalo cobre o paciente caído até o fim (o vídeo não pode voltar a "OK" no chão)
    fall = next(d for d in r.deviations if d.kind == "fall")
    label = next(lb for lb in s.labels if lb["kind"] == "fall")
    assert label["start_frame"] - s.fps * 0.5 <= fall.start_frame <= label["start_frame"]
    assert fall.end_frame >= label["end_frame"] - 1


def test_missing_frames_are_tolerated(analyzer, t0):
    s = syn.generate_squat_session(reps=5, seed=12)
    kp = s.keypoints.copy()
    kp[20:24] = np.nan  # detector perdeu a pessoa por 4 frames
    assert len(analyzer.analyze(kp, s.fps, "P", t0).reps) == 5


def test_surgical_rules(t0):
    an = SurgicalVideoAnalyzer()
    dets, _ = syn.generate_surgical_detections(scenario="retained_gauze", seed=2)
    kinds = {f.kind for f in an.analyze(dets, 5.0, "S", t0).findings}
    assert kinds == {"instrument_outside_field", "camera_occlusion", "gauze_count_mismatch"}
    dets, _ = syn.generate_surgical_detections(scenario="normal", seed=3)
    assert an.analyze(dets, 5.0, "S", t0).findings == []
