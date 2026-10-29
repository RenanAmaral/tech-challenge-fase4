"""Extração de features acústicas da fala.

Backend principal: Praat via ``praat-parselmouth`` (medidas clínicas padrão de
jitter, shimmer e HNR). Fallback puro em NumPy quando o Praat não está
instalado (medidas aproximadas, marcadas em ``backend``).

Features (todas independentes de idioma):
  * f0_mean, f0_sd_st — frequência fundamental e variação em semitons (monotonia)
  * jitter_pct, shimmer_pct, hnr_db — qualidade vocal (disartria, fadiga)
  * intensity_db — intensidade média relativa (fadiga → voz fraca)
  * syllable_rate — sílabas/s (bradilalia em fadiga/disartria)
  * rate_cv — irregularidade do ritmo (disartria atáxica)
  * pause_ratio, mean_pause_s, n_long_pauses — pausas (fadiga, dispneia)
  * syllables_per_phrase — frases curtas (dispneia)
  * breath_per_min — inspirações audíveis entre frases (dispneia)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import signal

try:  # pragma: no cover - depende do ambiente
    import parselmouth
    from parselmouth.praat import call

    HAS_PRAAT = True
except ImportError:  # pragma: no cover
    HAS_PRAAT = False

FRAME = 0.025
HOP = 0.010


@dataclass
class AcousticFeatures:
    duration_s: float
    f0_mean: float
    f0_sd_st: float
    jitter_pct: float
    shimmer_pct: float
    hnr_db: float
    intensity_db: float
    syllable_rate: float
    rate_cv: float
    pause_ratio: float
    mean_pause_s: float
    n_long_pauses: int
    syllables_per_phrase: float
    breath_per_min: float
    backend: str = "praat"

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    def vector(self, names: list[str]) -> np.ndarray:
        d = asdict(self)
        return np.array([float(d[n]) for n in names])


FEATURE_NAMES = [
    "f0_sd_st", "jitter_pct", "shimmer_pct", "hnr_db", "intensity_db", "syllable_rate",
    "rate_cv", "pause_ratio", "mean_pause_s", "syllables_per_phrase", "breath_per_min",
]


def load_audio(path: str | Path, sr: int = 16000) -> np.ndarray:
    """Lê WAV/FLAC/OGG em mono float32 no ``sr`` pedido."""
    import soundfile as sf

    y, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = signal.resample_poly(y, sr, file_sr).astype(np.float32)
    return y


def _frame_db(y: np.ndarray, sr: int) -> np.ndarray:
    fl, hp = int(FRAME * sr), int(HOP * sr)
    n = 1 + max(0, (len(y) - fl) // hp)
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    fr = y[idx] * np.hanning(fl)
    rms = np.sqrt(np.mean(fr**2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-9)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Intervalos [início, fim) onde ``mask`` é verdadeiro."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _numpy_pitch(y: np.ndarray, sr: int, fmin=75, fmax=500) -> tuple[np.ndarray, np.ndarray]:
    """f0 por autocorrelação normalizada; devolve (f0, força_periodicidade)."""
    fl, hp = int(0.04 * sr), int(HOP * sr)
    n = 1 + max(0, (len(y) - fl) // hp)
    f0 = np.zeros(n)
    strength = np.zeros(n)
    lo, hi = int(sr / fmax), int(sr / fmin)
    for i in range(n):
        x = y[i * hp : i * hp + fl]
        x = x - x.mean()
        if np.dot(x, x) < 1e-8:
            continue
        ac = np.correlate(x, x, "full")[fl - 1 :]
        ac = ac / (ac[0] + 1e-12)
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.45:
            f0[i] = sr / k
            strength[i] = ac[k]
    return f0, strength


def extract_features(y: np.ndarray, sr: int = 16000, backend: str = "auto") -> AcousticFeatures:
    """Calcula as features acústicas de uma gravação de fala."""
    use_praat = HAS_PRAAT and backend in ("auto", "praat")
    y = np.asarray(y, dtype=np.float64)
    duration = len(y) / sr
    db = _frame_db(y, sr)
    peak = np.percentile(db, 99)
    active = db > peak - 30  # frames com som (fala ou respiração)

    if use_praat:
        snd = parselmouth.Sound(y, sr)
        pitch = snd.to_pitch(time_step=HOP, pitch_floor=75, pitch_ceiling=500)
        f0_all = pitch.selected_array["frequency"]
        t_p = pitch.xs()
        voiced_p = f0_all > 0
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
        harm = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harm, "Get mean", 0, 0)
    else:
        f0_all, strength = _numpy_pitch(y, sr)
        t_p = np.arange(len(f0_all)) * HOP + 0.02
        voiced_p = f0_all > 0
        fv = f0_all[voiced_p]
        jitter = float(np.mean(np.abs(np.diff(1 / fv))) / np.mean(1 / fv) * 100) if len(fv) > 2 else 0.0
        r = np.clip(strength[voiced_p], 1e-3, 0.999)
        hnr = float(np.mean(10 * np.log10(r / (1 - r)))) if len(r) else 0.0
        fl = int(FRAME * sr)
        amps = np.array([np.max(np.abs(y[int(t * sr) : int(t * sr) + fl])) for t in t_p[voiced_p]])
        shimmer = float(np.mean(np.abs(np.diff(amps))) / (np.mean(amps) + 1e-9) * 100) if len(amps) > 2 else 0.0

    # alinhar vozeamento à grade de intensidade
    t_db = np.arange(len(db)) * HOP + FRAME / 2
    voiced = np.interp(t_db, t_p, voiced_p.astype(float), left=0, right=0) > 0.5

    f0 = f0_all[voiced_p]
    f0_mean = float(np.mean(f0)) if len(f0) else 0.0
    f0_sd_st = float(np.std(12 * np.log2(f0 / np.median(f0)))) if len(f0) > 2 else 0.0
    intensity = float(np.mean(db[active])) if active.any() else float(np.mean(db))

    # núcleos silábicos (de Jong & Wempe, 2009): picos de intensidade vozeados
    sm = signal.savgol_filter(db, 7, 2) if len(db) > 7 else db
    peaks, _ = signal.find_peaks(sm, prominence=2.0, distance=int(0.08 / HOP), height=peak - 25)
    peaks = peaks[voiced[peaks]]
    n_syll = len(peaks)

    # pausas = silêncios ≥ 250 ms entre trechos de fala
    speech = voiced | (active & (sm > peak - 20))
    speech_runs = _runs(speech)
    if speech_runs:
        first, last = speech_runs[0][0], speech_runs[-1][1]
        inner = ~speech[first:last]
        pauses = [(s + first, e + first) for s, e in _runs(inner) if (e - s) * HOP >= 0.25]
    else:
        pauses = []
    pause_durs = np.array([(e - s) * HOP for s, e in pauses])
    pause_ratio = float(pause_durs.sum() / duration) if duration else 0.0
    mean_pause = float(pause_durs.mean()) if len(pause_durs) else 0.0
    n_long = int(np.sum(pause_durs >= 0.6))
    n_phrases = len(pauses) + 1
    syll_per_phrase = n_syll / n_phrases

    # inspiração audível: trecho ativo, não vozeado, ≥ 250 ms, dentro de pausa
    unvoiced_active = active & ~voiced
    breaths = [r for r in _runs(unvoiced_active) if (r[1] - r[0]) * HOP >= 0.25]
    breath_per_min = len(breaths) / (duration / 60) if duration else 0.0

    intervals = np.diff(peaks) * HOP
    intervals = intervals[intervals < 0.6]  # só dentro de frases
    rate_cv = float(np.std(intervals) / np.mean(intervals)) if len(intervals) > 2 else 0.0

    return AcousticFeatures(
        duration_s=duration,
        f0_mean=f0_mean,
        f0_sd_st=f0_sd_st,
        jitter_pct=float(jitter),
        shimmer_pct=float(shimmer),
        hnr_db=float(hnr),
        intensity_db=intensity,
        syllable_rate=n_syll / duration if duration else 0.0,
        rate_cv=rate_cv,
        pause_ratio=pause_ratio,
        mean_pause_s=mean_pause,
        n_long_pauses=n_long,
        syllables_per_phrase=float(syll_per_phrase),
        breath_per_min=float(breath_per_min),
        backend="praat" if use_praat else "numpy",
    )
