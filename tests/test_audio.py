import soundfile as sf

from hospital_monitor.audio.acoustic import extract_features
from hospital_monitor.audio.clinical_nlp import analyze_text, extract_findings
from hospital_monitor.audio.pipeline import AudioPipeline
from hospital_monitor.audio.transcribe import SidecarTranscriber
from hospital_monitor.audio.voice_analyzer import ReferenceProfile, VoiceAnalyzer
from hospital_monitor.data import synthetic as syn


def _ref():
    return ReferenceProfile.from_features(
        [extract_features(syn.synthesize_voice("normal", seed=i, speaker_variation=True)) for i in range(6)])


def test_features_separate_conditions():
    n = extract_features(syn.synthesize_voice("normal", seed=1))
    d = extract_features(syn.synthesize_voice("dysarthria", seed=1))
    p = extract_features(syn.synthesize_voice("dyspnea", seed=1))
    assert d.jitter_pct > n.jitter_pct and d.hnr_db < n.hnr_db
    assert p.breath_per_min > n.breath_per_min and p.syllables_per_phrase < n.syllables_per_phrase


def test_voice_analyzer_flags_conditions():
    va = VoiceAnalyzer(_ref())
    assert va.assess(extract_features(syn.synthesize_voice("normal", seed=50))).main_condition is None
    assert va.assess(extract_features(syn.synthesize_voice("dysarthria", seed=51))).main_condition == "disartria"
    assert va.assess(extract_features(syn.synthesize_voice("dyspnea", seed=52))).main_condition == "dispneia"
    assert va.assess(extract_features(syn.synthesize_voice("fatigue", seed=53))).main_condition == "fadiga"


def test_negation():
    f = {x.term: x.negated for x in extract_findings("Não tenho falta de ar, mas sinto dor no peito.")}
    assert f == {"dispneia": True, "dor torácica": False}
    assert analyze_text("Nega febre e nega tosse.").anomaly_score == 0


def test_critical_terms_and_sentiment():
    a = analyze_text(syn.CONSULT_TRANSCRIPTS["dyspnea"])
    assert {"dispneia", "dor torácica"} <= set(a.critical_terms)
    assert a.sentiment["label"] == "NEG" and a.anomaly_score >= 0.9
    assert analyze_text(syn.CONSULT_TRANSCRIPTS["normal"]).anomaly_score == 0


def test_pipeline_end_to_end(tmp_path, t0):
    wav = tmp_path / "c.wav"
    sf.write(wav, syn.synthesize_voice("dysarthria", seed=7), 16000)
    wav.with_suffix(".txt").write_text(syn.CONSULT_TRANSCRIPTS["dysarthria"], encoding="utf-8")
    res = AudioPipeline(SidecarTranscriber(), VoiceAnalyzer(_ref())).analyze(wav, "P9", t0)
    kinds = {e.kind for e in res.events}
    assert kinds == {"voz_disartria", "relato_sintomas"}
    assert "disartria" in res.text.critical_terms
