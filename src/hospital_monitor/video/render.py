"""Renderização de esqueletos (frames anotados e vídeo MP4) com OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .pose import SKELETON

OK = (90, 170, 60)       # BGR
BAD = (40, 40, 220)
INK = (40, 40, 40)
# OpenCV não desenha acentos: rótulos ASCII
LABEL = {"shallow": "amplitude reduzida", "asymmetric": "assimetria", "valgus": "valgo dinamico",
         "trunk_lean": "tronco inclinado", "atypical": "padrao atipico", "fall": "queda"}


def project(kp_xyz: np.ndarray, size: int = 480, yaw_deg: float = 35.0) -> np.ndarray:
    """Projeta coordenadas 3D (m, y para cima) ou 2D normalizadas em pixels."""
    if kp_xyz.shape[-1] == 3:
        a = np.radians(yaw_deg)
        x = kp_xyz[:, 0] * np.cos(a) + kp_xyz[:, 2] * np.sin(a)
        y = kp_xyz[:, 1]
        scale = size * 0.72 / 1.7
        u = size / 2 + x * scale
        v = size * 0.92 - y * scale
    else:
        u = kp_xyz[:, 0] * size
        v = (1 - kp_xyz[:, 1]) * size
    return np.stack([u, v], axis=1)


def draw_frame(kp: np.ndarray, size: int = 480, bad: bool = False, label: str = "", sub: str = "") -> np.ndarray:
    img = np.full((size, size, 3), 248, np.uint8)
    cv2.line(img, (20, int(size * 0.92) + 4), (size - 20, int(size * 0.92) + 4), (200, 200, 200), 2)
    xyz = kp[:, :-1]
    if np.isnan(xyz).all():
        return img
    pts = project(np.nan_to_num(xyz), size)
    color = BAD if bad else OK
    for a, b in SKELETON:
        cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), color, 4, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, (int(p[0]), int(p[1])), 5, INK, -1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BAD if bad else INK, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
    return img


def render_session(kp: np.ndarray, fps: float, path: str | Path, deviations=(), size: int = 480) -> Path:
    """Gera MP4 do esqueleto; frames com desvio ficam em vermelho com o rótulo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for i, f in enumerate(kp):
        active = [d for d in deviations if d.start_frame <= i <= d.end_frame]
        lbl = LABEL.get(active[0].kind, active[0].kind).upper() if active else "OK"
        vw.write(draw_frame(f, size, bool(active), lbl, f"t = {i / fps:5.1f} s"))
    vw.release()
    return path


def keyframe_png(kp: np.ndarray, frame: int, path: str | Path, label: str, bad: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), draw_frame(kp[frame], 360, bad, label))
    return path
