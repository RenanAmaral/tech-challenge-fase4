"""Geradores de dados sintéticos com anomalias rotuladas (ground truth).

Usados na demo, nos testes e para medir precision/recall dos detectores. Cada
gerador é determinístico dado o ``seed`` e devolve os rótulos verdadeiros,
permitindo avaliação objetiva. Para dados reais, veja ``data/physionet.py``.

Nenhum dado aqui corresponde a pacientes reais (LGPD).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy import signal

# ----------------------------------------------------------------------------
# Sinais vitais
# ----------------------------------------------------------------------------

VITAL_COLUMNS = ["hr", "spo2", "sbp", "dbp", "rr", "temp"]


@dataclass
class VitalAnomaly:
    kind: str  # desaturation | tachycardia | hypotension | silent_deterioration | fever
    start_min: int
    duration_min: int


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    eps = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def generate_vitals(
    patient_id: str = "P001",
    start: datetime | None = None,
    hours: int = 24,
    anomalies: list[VitalAnomaly] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Série de sinais vitais a 1 amostra/min com anomalias injetadas.

    Retorna colunas: timestamp, patient_id, hr, spo2, sbp, dbp, rr, temp,
    label (0/1) e anomaly_type.
    """
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 10, 28, 0, 0, tzinfo=timezone.utc)
    n = hours * 60
    t = np.arange(n)
    hour = ((start.hour * 60 + start.minute + t) / 60.0) % 24
    circ = np.sin(2 * np.pi * (hour - 10) / 24)  # pico à tarde, vale de madrugada

    hr = 74 + 6 * circ + _ar1(n, 0.95, 1.0, rng)
    spo2 = 97.2 + _ar1(n, 0.9, 0.25, rng)
    sbp = 121 + 7 * circ + _ar1(n, 0.95, 1.2, rng)
    dbp = 78 + 4 * circ + _ar1(n, 0.95, 0.8, rng)
    rr = 15 + 1.0 * circ + _ar1(n, 0.9, 0.4, rng)
    temp = 36.7 + 0.25 * circ + _ar1(n, 0.98, 0.02, rng)

    label = np.zeros(n, dtype=int)
    atype = np.array([""] * n, dtype=object)

    for a in anomalies or []:
        s, e = a.start_min, min(n, a.start_min + a.duration_min)
        L = e - s
        ramp = np.clip(np.minimum(np.arange(L) / max(1, L * 0.2), (L - np.arange(L)) / max(1, L * 0.2)), 0, 1)
        if a.kind == "desaturation":
            spo2[s:e] -= 10.5 * ramp
            hr[s:e] += 18 * ramp
            rr[s:e] += 8 * ramp
        elif a.kind == "tachycardia":
            hr[s:e] += 62 * ramp
        elif a.kind == "hypotension":
            sbp[s:e] -= 40 * ramp
            dbp[s:e] -= 22 * ramp
            hr[s:e] += 20 * ramp
        elif a.kind == "silent_deterioration":
            # piora sutil e combinada: nenhum sinal cruza limiar sozinho
            hr[s:e] += 22 * ramp
            rr[s:e] += 6 * ramp
            spo2[s:e] -= 3.0 * ramp
            sbp[s:e] -= 12 * ramp
        elif a.kind == "fever":
            temp[s:e] += 2.0 * ramp
            hr[s:e] += 15 * ramp
        else:
            raise ValueError(f"anomalia desconhecida: {a.kind}")
        label[s:e] = 1
        atype[s:e] = a.kind

    df = pd.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=int(i)) for i in t],
            "patient_id": patient_id,
            "hr": np.round(hr, 1),
            "spo2": np.round(np.clip(spo2, 70, 100), 1),
            "sbp": np.round(sbp, 0),
            "dbp": np.round(dbp, 0),
            "rr": np.round(np.clip(rr, 4, 50), 1),
            "temp": np.round(temp, 2),
            "label": label,
            "anomaly_type": atype,
        }
    )
    return df


# ----------------------------------------------------------------------------
# Prescrições
# ----------------------------------------------------------------------------


def generate_prescriptions(
    patient_id: str = "P001",
    start: datetime | None = None,
    days: int = 3,
    anomalies: list[dict] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Histórico de prescrições com alterações normais e anômalas.

    ``anomalies``: lista de dicts com ``kind`` (decimal_error, above_max,
    duplication, abrupt_change) e ``at_hours``.
    """
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 10, 26, 8, 0, tzinfo=timezone.utc)
    base = [
        ("dipirona", 1000, 4, "mg", "IV"),
        ("omeprazol", 40, 1, "mg", "IV"),
        ("enoxaparina", 40, 1, "mg", "SC"),
        ("losartana", 50, 1, "mg", "VO"),
        ("ceftriaxona", 1000, 2, "mg", "IV"),
    ]
    rows = []
    for drug, dose, freq, unit, route in base:
        rows.append(dict(timestamp=start, drug=drug, dose=dose, freq_per_day=freq, unit=unit, route=route,
                         prescriber=f"CRM{rng.integers(1000, 9999)}", label=0, anomaly_type=""))
    # ajustes rotineiros diários (pequenas variações aceitáveis)
    for d in range(1, days):
        ts = start + timedelta(days=d, minutes=int(rng.integers(0, 90)))
        rows.append(dict(timestamp=ts, drug="losartana", dose=float(rng.choice([50, 50, 100])), freq_per_day=1,
                         unit="mg", route="VO", prescriber="CRM2211", label=0, anomaly_type=""))
        rows.append(dict(timestamp=ts + timedelta(minutes=5), drug="dipirona", dose=1000.0,
                         freq_per_day=int(rng.choice([3, 4])), unit="mg", route="IV", prescriber="CRM2211",
                         label=0, anomaly_type=""))
        if d == 1:
            rows.append(dict(timestamp=ts + timedelta(hours=2), drug="furosemida", dose=40.0, freq_per_day=1,
                             unit="mg", route="IV", prescriber="CRM3020", label=0, anomaly_type=""))

    for a in anomalies or []:
        ts = start + timedelta(hours=a["at_hours"])
        k = a["kind"]
        if k == "decimal_error":  # 40 mg virou 400 mg
            rows.append(dict(timestamp=ts, drug="enoxaparina", dose=400.0, freq_per_day=1, unit="mg", route="SC",
                             prescriber="CRM7781", label=1, anomaly_type=k))
        elif k == "above_max":
            rows.append(dict(timestamp=ts, drug="dipirona", dose=2000.0, freq_per_day=4, unit="mg", route="IV",
                             prescriber="CRM7781", label=1, anomaly_type=k))
        elif k == "duplication":  # segundo anticoagulante com enoxaparina ativa
            rows.append(dict(timestamp=ts, drug="heparina", dose=5000.0, freq_per_day=3, unit="UI", route="SC",
                             prescriber="CRM5512", label=1, anomaly_type=k))
        elif k == "abrupt_change":
            rows.append(dict(timestamp=ts, drug="furosemida", dose=200.0, freq_per_day=2, unit="mg", route="IV",
                             prescriber="CRM5512", label=1, anomaly_type=k))
        else:
            raise ValueError(k)

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.insert(1, "patient_id", patient_id)
    return df


# ----------------------------------------------------------------------------
# Movimentação durante a internação (acelerômetro / energia de movimento do vídeo)
# ----------------------------------------------------------------------------


def generate_movement(
    patient_id: str = "P001",
    start: datetime | None = None,
    hours: int = 48,
    anomalies: list[dict] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Contagem de atividade a cada 5 min (0–100).

    ``anomalies``: dicts com ``kind`` (immobility, night_agitation, fall),
    ``start_h`` e ``duration_h``.
    """
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 10, 27, 0, 0, tzinfo=timezone.utc)
    step = 5
    n = hours * 60 // step
    ts = [start + timedelta(minutes=step * i) for i in range(n)]
    hour = np.array([(t.hour + t.minute / 60) for t in ts])
    awake = (hour >= 7) & (hour < 22)
    base = np.where(awake, 22.0, 4.0)
    # reposicionamento da enfermagem a cada 2 h durante o dia (picos)
    act = base + rng.gamma(2.0, np.where(awake, 6.0, 1.5))
    act += np.where(awake & (rng.random(n) < 0.08), rng.uniform(20, 40, n), 0)
    label = np.zeros(n, dtype=int)
    atype = np.array([""] * n, dtype=object)
    for a in anomalies or []:
        s = int(a["start_h"] * 60 / step)
        e = min(n, s + max(1, int(a.get("duration_h", 0.1) * 60 / step)))
        if a["kind"] == "immobility":
            act[s:e] = rng.uniform(0, 1.2, e - s)
        elif a["kind"] == "night_agitation":
            act[s:e] = rng.uniform(35, 70, e - s)
        elif a["kind"] == "fall":
            act[s] = 98.0
            e = min(n, s + 4)
            act[s + 1 : e] = rng.uniform(0, 1.0, e - s - 1)
        label[s:e] = 1
        atype[s:e] = a["kind"]
    return pd.DataFrame({"timestamp": ts, "patient_id": patient_id, "activity": np.round(np.clip(act, 0, 100), 1),
                         "label": label, "anomaly_type": atype})


# ----------------------------------------------------------------------------
# Vídeo: keypoints 3D de sessão de fisioterapia (agachamento)
# ----------------------------------------------------------------------------

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

THIGH, SHIN = 0.45, 0.45


def _knee_ik(hip: np.ndarray, ankle: np.ndarray) -> np.ndarray:
    """Posição do joelho por cinemática inversa de 2 segmentos (joelho à frente, +z)."""
    v = hip - ankle
    dist = np.linalg.norm(v)
    dist = min(dist, THIGH + SHIN - 1e-6)
    # ângulo no tornozelo entre a reta tornozelo-quadril e a tíbia (lei dos cossenos)
    cos_a = (SHIN**2 + dist**2 - THIGH**2) / (2 * SHIN * dist)
    a = np.arccos(np.clip(cos_a, -1, 1))
    u = v / np.linalg.norm(v)
    # vetor perpendicular no plano sagital apontando para frente (+z)
    fwd = np.array([0.0, 0.0, 1.0])
    perp = fwd - np.dot(fwd, u) * u
    if np.linalg.norm(perp) < 1e-9:
        perp = np.array([0.0, 0.0, 1.0])
    perp /= np.linalg.norm(perp)
    return ankle + SHIN * (np.cos(a) * u + np.sin(a) * perp)


@dataclass
class ExerciseSession:
    keypoints: np.ndarray  # (T, 17, 4) → x, y, z (m, y para cima), confiança
    fps: float
    labels: list[dict] = field(default_factory=list)  # [{kind, start_frame, end_frame}]


def generate_squat_session(
    reps: int = 5,
    fps: float = 15.0,
    deviation: str | None = None,
    deviation_reps: tuple[int, ...] = (),
    fall_at_rep: int | None = None,
    seed: int = 0,
) -> ExerciseSession:
    """Gera keypoints 3D (formato COCO-17) de agachamentos.

    deviation: ``shallow`` (amplitude reduzida), ``asymmetric`` (perna direita
    flexiona menos), ``valgus`` (joelhos para dentro), ``trunk_lean``
    (inclinação excessiva do tronco). Aplicada às repetições de
    ``deviation_reps`` (índices a partir de 0). ``fall_at_rep`` simula queda.
    """
    rng = np.random.default_rng(seed)
    period = 2.6
    rest = 0.6
    frames: list[np.ndarray] = []
    labels: list[dict] = []
    ankle_y = 0.08
    stand_h = ankle_y + THIGH + SHIN - 0.008

    def pose(dl: float, dr: float, lean: float, valgus: float) -> np.ndarray:
        kp = np.zeros((17, 3))
        la = np.array([0.12, ankle_y, 0.0])
        ra = np.array([-0.12, ankle_y, 0.0])
        lh = np.array([0.10, stand_h - 0.42 * dl, -0.22 * dl])
        rh = np.array([-0.10, stand_h - 0.42 * dr, -0.22 * dr])
        lk = _knee_ik(lh, la)
        rk = _knee_ik(rh, ra)
        lk[0] -= valgus * 0.09 * dl  # joelho esquerdo migra para medial (x→0)
        rk[0] += valgus * 0.09 * dr
        pelvis = (lh + rh) / 2
        trunk = 0.52
        phi = np.radians(lean)
        up = np.array([0.0, np.cos(phi), np.sin(phi)])
        sh_c = pelvis + trunk * up
        ls, rs = sh_c + [0.18, 0, 0], sh_c + [-0.18, 0, 0]
        head = sh_c + 0.22 * up
        arm_fwd = np.array([0, -0.05, 0.28]) * (0.3 + 0.7 * max(dl, dr))
        le, re = ls + [0.02, -0.25, 0] + arm_fwd * 0.6, rs + [-0.02, -0.25, 0] + arm_fwd * 0.6
        lw, rw = le + [0, -0.2, 0] + arm_fwd, re + [0, -0.2, 0] + arm_fwd
        kp[0] = head + [0, 0, 0.08]
        kp[1], kp[2] = head + [0.03, 0.03, 0.07], head + [-0.03, 0.03, 0.07]
        kp[3], kp[4] = head + [0.07, 0.01, 0], head + [-0.07, 0.01, 0]
        kp[5], kp[6], kp[7], kp[8], kp[9], kp[10] = ls, rs, le, re, lw, rw
        kp[11], kp[12], kp[13], kp[14], kp[15], kp[16] = lh, rh, lk, rk, la, ra
        return kp

    t_frame = 0
    for r in range(reps):
        dev = deviation if r in deviation_reps else None
        depth = 1.0 * rng.uniform(0.95, 1.05)
        n_rep = int(period * fps * rng.uniform(0.95, 1.05))
        start_f = t_frame
        for i in range(n_rep):
            d = 0.5 * (1 - np.cos(2 * np.pi * i / n_rep)) * depth
            dl = dr = d
            lean, valg = 25 * d, 0.0
            if dev == "shallow":
                dl = dr = d * 0.45
                lean = 25 * dl
            elif dev == "asymmetric":
                dr = d * 0.55
            elif dev == "valgus":
                valg = 1.0
            elif dev == "trunk_lean":
                lean = 62 * d
            frames.append(pose(min(dl, 1.0), min(dr, 1.0), lean, valg))
            t_frame += 1
        if dev:
            labels.append({"kind": dev, "rep": r, "start_frame": start_f, "end_frame": t_frame - 1})
        if fall_at_rep is not None and r == fall_at_rep:
            # queda: corpo gira em torno dos tornozelos até a horizontal em 0,6 s
            base = pose(0.3, 0.3, 10, 0)
            n_fall = int(0.6 * fps)
            fall_start = t_frame
            for i in range(n_fall + int(3 * fps)):
                ang = np.radians(min(1.0, i / n_fall) * 88)
                rot = np.array([[1, 0, 0], [0, np.cos(ang), np.sin(ang)], [0, -np.sin(ang), np.cos(ang)]])
                piv = np.array([0, ankle_y, 0])
                kp = (base - piv) @ rot.T + piv
                kp[:, 1] = np.maximum(kp[:, 1], 0.03)
                frames.append(kp)
                t_frame += 1
            labels.append({"kind": "fall", "rep": r, "start_frame": fall_start, "end_frame": t_frame - 1})
            break
        for _ in range(int(rest * fps)):
            frames.append(pose(0, 0, 0, 0))
            t_frame += 1

    xyz = np.stack(frames)
    xyz += rng.normal(0, 0.004, xyz.shape)
    conf = np.clip(rng.normal(0.93, 0.03, xyz.shape[:2] + (1,)), 0.5, 1.0)
    return ExerciseSession(np.concatenate([xyz, conf], axis=-1), fps, labels)


# ----------------------------------------------------------------------------
# Vídeo cirúrgico: detecções por frame (saída típica do YOLOv8)
# ----------------------------------------------------------------------------


def generate_surgical_detections(
    minutes: float = 6.0,
    fps: float = 5.0,
    scenario: str = "retained_gauze",
    seed: int = 0,
) -> tuple[list[list[dict]], list[dict]]:
    """Simula detecções do YOLOv8 em cirurgia gravada.

    Classes: pinca, tesoura, bisturi, porta_agulha, gaze, mao. Bboxes
    normalizadas (x1, y1, x2, y2). Cenários: ``normal`` ou
    ``retained_gauze`` (gaze some do campo + bisturi fora do campo estéril +
    obstrução da câmera).
    """
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * fps)
    labels: list[dict] = []

    def box(cx, cy, w=0.06, h=0.04):
        return [round(cx - w / 2, 4), round(cy - h / 2, 4), round(cx + w / 2, 4), round(cy + h / 2, 4)]

    tray = {"pinca": (0.30, 0.30), "tesoura": (0.38, 0.30), "bisturi": (0.46, 0.30), "porta_agulha": (0.54, 0.30)}
    gauze_pos = [(0.30 + 0.07 * i, 0.70) for i in range(5)]
    gauze_alive = [True] * 5
    frames: list[list[dict]] = []

    out_start, out_end = int(n * 0.35), int(n * 0.35 + 45 * fps)
    occ_start, occ_end = int(n * 0.6), int(n * 0.6 + 9 * fps)
    gauze_lost_at = int(n * 0.5)
    if scenario == "retained_gauze":
        labels += [
            {"kind": "instrument_outside_field", "start_frame": out_start, "end_frame": out_end},
            {"kind": "camera_occlusion", "start_frame": occ_start, "end_frame": occ_end},
            {"kind": "gauze_count_mismatch", "start_frame": n - 1, "end_frame": n - 1},
        ]

    for f in range(n):
        dets: list[dict] = []
        if scenario == "retained_gauze" and occ_start <= f < occ_end:
            frames.append(dets)
            continue
        if scenario == "retained_gauze" and f == gauze_lost_at:
            gauze_alive[2] = False  # gaze ficou na cavidade
        for name, (x, y) in tray.items():
            cx, cy = x + rng.normal(0, 0.004), y + rng.normal(0, 0.004)
            if name == "bisturi" and scenario == "retained_gauze" and out_start <= f < out_end:
                cx, cy = 0.92, 0.12  # fora do campo estéril
            if rng.random() > 0.08:  # falha de detecção ocasional
                dets.append({"cls": name, "conf": round(float(rng.uniform(0.6, 0.95)), 2), "bbox": box(cx, cy)})
        for i, (x, y) in enumerate(gauze_pos):
            if gauze_alive[i] and rng.random() > 0.1:
                dets.append({"cls": "gaze", "conf": round(float(rng.uniform(0.55, 0.9)), 2),
                             "bbox": box(x + rng.normal(0, 0.003), y, 0.05, 0.05)})
        for hx in (0.45, 0.6):
            if rng.random() > 0.15:
                dets.append({"cls": "mao", "conf": 0.9, "bbox": box(hx + rng.normal(0, 0.02), 0.5 + rng.normal(0, 0.02), 0.1, 0.1)})
        frames.append(dets)
    return frames, labels


# ----------------------------------------------------------------------------
# Áudio: voz sintética com marcadores de fadiga, disartria e dispneia
# ----------------------------------------------------------------------------

VOICE_PROFILES = {
    #             f0   into  jitter shimmer noise  rate  pause  amp  dur_cv phrase breath
    "normal":     (125, 0.12, 0.003, 0.025, 0.010, 4.8, 0.30, 0.8, 0.10, 7, False),
    "fatigue":    (108, 0.05, 0.007, 0.050, 0.045, 3.0, 1.10, 0.35, 0.15, 5, False),
    "dysarthria": (118, 0.03, 0.022, 0.085, 0.080, 2.6, 0.60, 0.6, 0.45, 5, False),
    "dyspnea":    (122, 0.08, 0.006, 0.045, 0.040, 4.2, 0.70, 0.55, 0.12, 2, True),
}

_VOWELS = [(730, 1090, 2440), (530, 1840, 2480), (270, 2290, 3010), (570, 840, 2410), (300, 870, 2240)]


def _resonator(x: np.ndarray, f: float, bw: float, sr: int) -> np.ndarray:
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * f / sr
    b = [1 - r]
    a = [1, -2 * r * np.cos(theta), r * r]
    return signal.lfilter(b, a, x)


def synthesize_voice(
    condition: str = "normal",
    seconds: float = 12.0,
    sr: int = 16000,
    seed: int = 0,
    speaker_variation: bool = False,
) -> np.ndarray:
    """Fala sintética (fonte-filtro): trem de pulsos glotais + formantes.

    Os perfis controlam f0, entonação, jitter, shimmer, ruído de aspiração,
    taxa de sílabas e pausas — os mesmos marcadores medidos em
    ``audio/acoustic.py``. Serve para validar o pipeline sem gravar pacientes.
    ``speaker_variation`` perturba os parâmetros (±15%) e sorteia voz
    masculina/feminina, simulando falantes diferentes para treino de modelos.
    """
    rng = np.random.default_rng(seed)
    f0b, into, jit, shim, noise, rate, pause, amp, dcv, phrase, breath = VOICE_PROFILES[condition]
    if speaker_variation:
        v = lambda x: x * rng.uniform(0.85, 1.15)  # noqa: E731
        f0b = v(f0b) * (1.75 if rng.random() < 0.5 else 1.0)
        into, jit, shim, noise, rate, pause, amp, dcv = map(v, (into, jit, shim, noise, rate, pause, amp, dcv))
        phrase = max(2, int(round(v(phrase))))
    out: list[np.ndarray] = []
    total = 0
    syll_in_phrase = 0
    target = int(seconds * sr)
    phrase_pos = 0.0
    while total < target:
        dur = max(0.08, rng.normal(1.0 / rate, dcv / rate)) * 0.8
        n = int(dur * sr)
        # contorno de f0 com declinação ao longo da frase e entonação
        f0 = f0b * (1 + into * np.sin(np.pi * phrase_pos) - 0.06 * phrase_pos)
        exc = np.zeros(n)
        pos = 0.0
        while pos < n:
            idx = int(pos)
            a = amp * (1 + rng.normal(0, shim))
            exc[idx] += a
            period = sr / f0 * (1 + rng.normal(0, jit))
            pos += max(period, sr / 500)
        # pulso glotal suavizado + inclinação espectral
        exc = signal.lfilter([1.0], [1, -0.96], exc)
        exc = signal.lfilter([1.0], [1, -0.8], exc) * 0.05
        env = np.minimum(1, np.minimum(np.arange(n) / (0.02 * sr), (n - np.arange(n)) / (0.03 * sr)))
        exc = exc * env + rng.normal(0, noise, n) * env * amp
        F = _VOWELS[rng.integers(len(_VOWELS))]
        y = sum(_resonator(exc, f, bw, sr) * g for f, bw, g in zip(F, (80, 100, 140), (1.0, 0.6, 0.3)))
        cons = rng.normal(0, 0.02 * amp, int(0.045 * sr))  # consoante (fricativa curta)
        cons = signal.lfilter(*signal.butter(2, 3000 / (sr / 2), "high"), cons)
        out += [cons, y]
        total += len(cons) + n
        syll_in_phrase += 1
        phrase_pos = syll_in_phrase / phrase
        if syll_in_phrase >= phrase:
            syll_in_phrase = 0
            phrase_pos = 0.0
            gap = max(0.15, rng.normal(pause, 0.15 * pause))
            sil = rng.normal(0, 0.0008, int(gap * sr))
            if breath:  # inspiração audível entre frases curtas
                b = rng.normal(0, 0.05, int(0.45 * sr))
                b = signal.lfilter(*signal.butter(2, [400 / (sr / 2), 2500 / (sr / 2)], "band"), b)
                b *= np.hanning(len(b))
                sil = np.concatenate([sil[: len(sil) // 2], b, sil[len(sil) // 2 :]])
            out.append(sil)
            total += len(sil)
        else:
            gap = rng.uniform(0.01, 0.05)
            out.append(rng.normal(0, 0.0008, int(gap * sr)))
            total += int(gap * sr)
    y = np.concatenate(out)[:target]
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9 * (amp / 0.8 if amp < 0.8 else 1.0)
    return y.astype(np.float32)


CONSULT_TRANSCRIPTS = {
    "normal": (
        "Bom dia, doutor. Estou me sentindo bem melhor hoje. Não tenho dor no peito nem falta de ar. "
        "Dormi bem e consegui caminhar no corredor com a fisioterapeuta. Estou animado para ir para casa."
    ),
    "dyspnea": (
        "Doutor, desde a madrugada estou com muita falta de ar e um cansaço muito forte. "
        "Sinto uma pressão no peito quando tento levantar e fico tonto. Estou com medo, não consigo respirar direito."
    ),
    "fatigue": (
        "Estou muito cansado, doutor, uma fraqueza que não passa. Não tenho febre. "
        "Dormi mal e não consigo terminar os exercícios da fisioterapia."
    ),
    "dysarthria": (
        "Minha esposa disse que minha fala está enrolada desde ontem à noite. "
        "Sinto uma fraqueza no braço direito e o rosto meio dormente. Não tenho febre nem dor de cabeça."
    ),
}
