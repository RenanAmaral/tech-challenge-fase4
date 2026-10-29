"""Análise de sessões de fisioterapia (agachamento) a partir de keypoints.

Fluxo:
  1. limpeza (interpolação de frames perdidos + suavização Savitzky-Golay);
  2. ângulos articulares, valgo dinâmico e inclinação do tronco;
  3. segmentação automática das repetições pelos vales do ângulo do joelho;
  4. por repetição: amplitude (ROM), assimetria E/D, valgo, tronco e distância
     DTW contra a execução de referência (padrão esperado);
  5. detecção de queda (queda rápida do centro de massa + corpo horizontal);
  6. desvios → eventos de anomalia + dados para o relatório automático.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from ..events import Event, Modality
from . import kinematics as K

DEVIATION_TEXT = {
    "shallow": "amplitude de movimento reduzida (agachamento raso)",
    "asymmetric": "assimetria entre membros (descarga de peso desigual)",
    "valgus": "valgo dinâmico de joelho (joelho desloca para dentro)",
    "trunk_lean": "inclinação excessiva do tronco (compensação)",
    "atypical": "padrão de movimento diferente da referência (DTW)",
    "fall": "queda do paciente durante a sessão",
}
# identificador curto em pt-BR (tipo do evento e rótulo nos quadros)
DEVIATION_SLUG = {"shallow": "amplitude_reduzida", "asymmetric": "assimetria", "valgus": "valgo_dinamico",
                  "trunk_lean": "inclinacao_tronco", "atypical": "padrao_atipico", "fall": "queda"}


@dataclass
class ExerciseReference:
    """Padrão esperado do exercício, extraído de uma execução correta."""

    curve: list[list[float]]  # média das repetições reamostradas: (50, 3) joelhoE, joelhoD, tronco
    rom: float
    trunk_max: float
    dtw_mean: float
    dtw_std: float
    n_reps: int

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ExerciseReference":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class Thresholds:
    rom_ratio_min: float = 0.70  # ROM < 70% da referência → raso
    asym_deg: float = 15.0
    valgus_cm: float = 4.0
    trunk_extra_deg: float = 20.0
    dtw_sigma: float = 4.0  # DTW > média + 4σ da referência → atípico
    dtw_min: float = 8.0  # piso em graus (referências muito homogêneas)
    fall_drop_m: float = 0.30
    fall_window_s: float = 1.0
    fall_aspect: float = 1.2


@dataclass
class RepMetrics:
    rep: int
    start_frame: int
    end_frame: int
    rom: float
    min_knee: float
    asym_max: float
    valgus_max: float
    trunk_max: float
    dtw: float


@dataclass
class Deviation:
    kind: str
    rep: int | None
    start_frame: int
    end_frame: int
    score: float
    measured: float
    expected: str
    description: str


@dataclass
class ExerciseResult:
    patient_id: str
    fps: float
    n_frames: int
    reps: list[RepMetrics]
    deviations: list[Deviation]
    angles: dict[str, np.ndarray] = field(repr=False)
    events: list[Event] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "duration_s": round(self.n_frames / self.fps, 1),
            "n_reps": len(self.reps),
            "n_deviations": len(self.deviations),
            "reps": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(r).items()} for r in self.reps],
            "deviations": [asdict(d) | {"score": round(d.score, 3), "measured": round(d.measured, 2)} for d in self.deviations],
            "events": [e.to_dict() for e in self.events],
        }


def _prep(kp: np.ndarray) -> np.ndarray:
    kp = K.interpolate_missing(np.asarray(kp, dtype=float))
    kp[..., :-1] = K.smooth(kp[..., :-1])
    return kp


def segment_reps(knee: np.ndarray, fps: float, prominence: float = 25.0) -> list[tuple[int, int]]:
    """Repetições = vales do ângulo do joelho; limites nos picos (em pé) vizinhos."""
    valleys, _ = find_peaks(-knee, prominence=prominence, distance=int(0.8 * fps))
    peaks, _ = find_peaks(knee, distance=int(0.3 * fps))
    standing = np.concatenate([[0], peaks, [len(knee) - 1]])
    reps = []
    for v in valleys:
        s = standing[standing < v].max()
        e = standing[standing > v].min()
        reps.append((int(s), int(e)))
    return reps


def _rep_curve(ang: dict[str, np.ndarray], s: int, e: int) -> np.ndarray:
    sig = np.stack([ang["knee_l"][s : e + 1], ang["knee_r"][s : e + 1], ang["trunk_lean"][s : e + 1]], axis=1)
    return K.resample(sig, 50)


def build_reference(kp: np.ndarray, fps: float) -> ExerciseReference:
    kp = _prep(kp)
    ang = K.joint_angles(kp)
    knee = (ang["knee_l"] + ang["knee_r"]) / 2
    reps = segment_reps(knee, fps)
    if not reps:
        raise ValueError("nenhuma repetição encontrada na sessão de referência")
    curves = np.stack([_rep_curve(ang, s, e) for s, e in reps])
    mean_curve = curves.mean(axis=0)
    dtws = [K.dtw_distance(c, mean_curve, band=10) for c in curves]
    roms = [np.ptp(knee[s : e + 1]) for s, e in reps]
    trunk = [ang["trunk_lean"][s : e + 1].max() for s, e in reps]
    return ExerciseReference(mean_curve.round(3).tolist(), float(np.mean(roms)), float(np.mean(trunk)),
                             float(np.mean(dtws)), float(np.std(dtws)), len(reps))


def _score(excess_ratio: float) -> float:
    """Converte o quanto o limiar foi excedido em escore 0,5–0,75.

    Desvios de execução são achados de *atenção* (orientar o paciente, ajustar
    o exercício); só a queda é tratada como evento crítico (0,95).
    """
    return float(0.5 + 0.25 * np.tanh(max(0.0, excess_ratio)))


class ExerciseAnalyzer:
    def __init__(self, reference: ExerciseReference, thresholds: Thresholds | None = None):
        self.ref = reference
        self.th = thresholds or Thresholds()
        self.ref_curve = np.asarray(reference.curve)

    def detect_fall(self, kp: np.ndarray, fps: float) -> Deviation | None:
        com = K.center_of_mass(kp)[:, 1]
        aspect = K.body_aspect(kp)
        w = max(1, int(self.th.fall_window_s * fps))
        drop = com[:-w] - com[w:]
        cand = np.where(drop > self.th.fall_drop_m)[0]
        for c in cand:
            after = aspect[c + w : c + w + int(fps)]
            if len(after) and np.median(after) > self.th.fall_aspect:
                return Deviation("fall", None, int(c), int(min(len(com) - 1, c + w + int(fps))), 0.95,
                                 float(drop[c]), f"queda > {self.th.fall_drop_m} m em {self.th.fall_window_s} s",
                                 DEVIATION_TEXT["fall"])
        return None

    def analyze(self, kp: np.ndarray, fps: float, patient_id: str, start_time: datetime) -> ExerciseResult:
        kp = _prep(kp)
        ang = K.joint_angles(kp)
        knee = (ang["knee_l"] + ang["knee_r"]) / 2
        th, ref = self.th, self.ref
        dtw_thr = max(th.dtw_min, ref.dtw_mean + th.dtw_sigma * ref.dtw_std)
        reps: list[RepMetrics] = []
        devs: list[Deviation] = []
        for i, (s, e) in enumerate(segment_reps(knee, fps)):
            seg = slice(s, e + 1)
            rom = float(np.ptp(knee[seg]))
            asym = float(np.max(np.abs(ang["knee_l"][seg] - ang["knee_r"][seg])))
            valg = float(max(ang.get("valgus_l", np.zeros_like(knee))[seg].max(),
                             ang.get("valgus_r", np.zeros_like(knee))[seg].max()))
            trunk = float(ang["trunk_lean"][seg].max())
            dtw = K.dtw_distance(_rep_curve(ang, s, e), self.ref_curve, band=10)
            reps.append(RepMetrics(i, s, e, rom, float(knee[seg].min()), asym, valg, trunk, dtw))
            if rom < th.rom_ratio_min * ref.rom:
                devs.append(Deviation("shallow", i, s, e, _score((th.rom_ratio_min * ref.rom - rom) / (0.3 * ref.rom)),
                                      rom, f"≥ {th.rom_ratio_min * ref.rom:.0f}° de ROM", DEVIATION_TEXT["shallow"]))
            if asym > th.asym_deg:
                devs.append(Deviation("asymmetric", i, s, e, _score((asym - th.asym_deg) / th.asym_deg), asym,
                                      f"≤ {th.asym_deg:.0f}° de diferença E/D", DEVIATION_TEXT["asymmetric"]))
            if valg > th.valgus_cm:
                devs.append(Deviation("valgus", i, s, e, _score((valg - th.valgus_cm) / th.valgus_cm), valg,
                                      f"≤ {th.valgus_cm:.0f} cm medial", DEVIATION_TEXT["valgus"]))
            if trunk > ref.trunk_max + th.trunk_extra_deg:
                lim = ref.trunk_max + th.trunk_extra_deg
                devs.append(Deviation("trunk_lean", i, s, e, _score((trunk - lim) / th.trunk_extra_deg), trunk,
                                      f"≤ {lim:.0f}° de inclinação", DEVIATION_TEXT["trunk_lean"]))
            already = any(d.rep == i for d in devs)
            if dtw > dtw_thr and not already:
                devs.append(Deviation("atypical", i, s, e, _score((dtw - dtw_thr) / dtw_thr), dtw,
                                      f"DTW ≤ {dtw_thr:.1f}°", DEVIATION_TEXT["atypical"]))
        fall = self.detect_fall(kp, fps)
        if fall:
            devs.append(fall)

        events: list[Event] = []
        by_kind: dict[str, list[Deviation]] = {}
        for d in devs:
            by_kind.setdefault(d.kind, []).append(d)
        for kind, ds in by_kind.items():
            worst = max(ds, key=lambda d: d.score)
            reps_txt = sorted({d.rep + 1 for d in ds if d.rep is not None})
            desc = DEVIATION_TEXT[kind].capitalize()
            desc += f" em {len(reps_txt)} de {len(reps)} repetições ({', '.join(map(str, reps_txt))})." if reps_txt \
                else f" aos {worst.start_frame / fps:.1f} s."
            events.append(Event(
                patient_id=patient_id, modality=Modality.VIDEO,
                kind="queda" if kind == "fall" else f"exercicio_{DEVIATION_SLUG[kind]}",
                anomaly_score=worst.score, timestamp=start_time + timedelta(seconds=worst.start_frame / fps),
                source="video.exercise_analyzer", description=desc,
                evidence={"reps": reps_txt, "measured": round(worst.measured, 2), "expected": worst.expected,
                          "frames": [worst.start_frame, worst.end_frame]},
            ))
        return ExerciseResult(patient_id, fps, len(kp), reps, devs, ang, events)
