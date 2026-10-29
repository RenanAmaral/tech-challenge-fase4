"""Cinemática a partir de keypoints: ângulos articulares, simetria, DTW."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .pose import COCO


def _xyz(kp: np.ndarray) -> np.ndarray:
    """Remove a coluna de confiança: (T, 17, D+1) → (T, 17, D)."""
    return kp[..., :-1]


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ângulo ABC em graus, vetorizado sobre o tempo."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, -1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def interpolate_missing(x: np.ndarray) -> np.ndarray:
    """Preenche NaN (frames sem detecção) por interpolação linear no tempo."""
    x = x.copy()
    flat = x.reshape(len(x), -1)
    t = np.arange(len(x))
    for j in range(flat.shape[1]):
        col = flat[:, j]
        m = np.isnan(col)
        if m.all():
            continue
        if m.any():
            col[m] = np.interp(t[m], t[~m], col[~m])
    return flat.reshape(x.shape)


def smooth(x: np.ndarray, window: int = 7) -> np.ndarray:
    if len(x) <= window:
        return x
    return savgol_filter(x, window, 2, axis=0)


def joint_angles(kp: np.ndarray) -> dict[str, np.ndarray]:
    """Ângulos de joelho, quadril e inclinação do tronco (graus) por frame."""
    p = _xyz(kp)
    J = lambda n: p[:, COCO[n]]  # noqa: E731
    knee_l = angle(J("left_hip"), J("left_knee"), J("left_ankle"))
    knee_r = angle(J("right_hip"), J("right_knee"), J("right_ankle"))
    hip_l = angle(J("left_shoulder"), J("left_hip"), J("left_knee"))
    hip_r = angle(J("right_shoulder"), J("right_hip"), J("right_knee"))
    sh = (J("left_shoulder") + J("right_shoulder")) / 2
    pel = (J("left_hip") + J("right_hip")) / 2
    trunk = sh - pel
    up = np.zeros_like(trunk)
    up[:, 1] = 1.0
    trunk_lean = angle(pel + up, pel, pel + trunk)  # 0° = tronco ereto
    out = {"knee_l": knee_l, "knee_r": knee_r, "hip_l": hip_l, "hip_r": hip_r, "trunk_lean": trunk_lean}
    if p.shape[-1] == 3:  # valgo dinâmico só com coordenadas 3D (plano frontal)
        out["valgus_l"] = _valgus(J("left_hip"), J("left_knee"), J("left_ankle"), side=+1)
        out["valgus_r"] = _valgus(J("right_hip"), J("right_knee"), J("right_ankle"), side=-1)
    return out


def _valgus(hip, knee, ankle, side: int) -> np.ndarray:
    """Deslocamento medial do joelho (cm) em relação à reta quadril–tornozelo."""
    t = 0.5
    line_x = hip[:, 0] + t * (ankle[:, 0] - hip[:, 0])
    return side * (line_x - knee[:, 0]) * 100


def center_of_mass(kp: np.ndarray) -> np.ndarray:
    p = _xyz(kp)
    idx = [COCO[n] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    return np.nanmean(p[:, idx], axis=1)


def body_aspect(kp: np.ndarray) -> np.ndarray:
    """Razão largura/altura da caixa do corpo no plano da imagem (>1 ≈ deitado)."""
    p = _xyz(kp)
    rng = lambda a: np.nanmax(a, axis=1) - np.nanmin(a, axis=1)  # noqa: E731
    w = np.hypot(rng(p[..., 0]), rng(p[..., 2])) if p.shape[-1] == 3 else rng(p[..., 0])
    h = rng(p[..., 1])
    return w / (h + 1e-9)


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int | None = None) -> float:
    """Dynamic Time Warping (distância euclidiana local), normalizada pelo caminho.

    a, b: (n, d) ou (n,). ``band``: janela de Sakoe-Chiba.
    """
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    n, m = len(a), len(b)
    band = max(band or max(n, m), abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    L = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(m, i + band)
        for j in range(lo, hi + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            choices = (D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            k = int(np.argmin(choices))
            D[i, j] = cost + choices[k]
            L[i, j] = 1 + (L[i - 1, j], L[i, j - 1], L[i - 1, j - 1])[k]
    return float(D[n, m] / max(1, L[n, m]))


def resample(x: np.ndarray, n: int = 50) -> np.ndarray:
    t0 = np.linspace(0, 1, len(x))
    t1 = np.linspace(0, 1, n)
    if x.ndim == 1:
        return np.interp(t1, t0, x)
    return np.stack([np.interp(t1, t0, x[:, j]) for j in range(x.shape[1])], axis=1)
