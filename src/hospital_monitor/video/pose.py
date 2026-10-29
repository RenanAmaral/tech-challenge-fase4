"""Estimativa de pose: vídeo → keypoints no formato COCO-17.

Backends intercambiáveis (mesma saída ``(T, 17, D+1)``: coordenadas + confiança):

* ``MediaPipePoseEstimator`` — MediaPipe Pose Landmarker (CPU, gratuito); usa
  os *world landmarks* 3D em metros, o que permite medir assimetria e valgo.
* ``YoloPoseEstimator`` — Ultralytics YOLOv8-pose (2D em pixels).
* ``load_keypoints`` — lê keypoints já extraídos (JSON/NPY), por exemplo do
  OpenPose (formato COCO), ou gerados por ``data/synthetic.py``.

Recomendação de gravação para backends 2D: câmera lateral à altura do quadril.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

COCO = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
# índice MediaPipe (33 pontos) para cada ponto COCO-17
MEDIAPIPE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
MP_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")


def read_video_frames(path: str | Path, target_fps: float | None = 15.0):
    """Gera (índice, timestamp_s, frame_bgr) com reamostragem opcional de fps."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / target_fps))) if target_fps else 1
    i = 0
    out_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield out_i, i / fps, frame
            out_i += 1
        i += 1
    cap.release()


class MediaPipePoseEstimator:  # pragma: no cover - exige modelo baixado
    def __init__(self, model_path: str | Path = "models/pose_landmarker_lite.task", world: bool = True):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(model_path)
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            log.info("baixando modelo MediaPipe para %s", model_path)
            urllib.request.urlretrieve(MP_MODEL_URL, model_path)
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
        )
        self._mp = mp
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self.world = world

    def estimate_video(self, path: str | Path, fps: float = 15.0) -> tuple[np.ndarray, float]:
        frames = []
        for _, t, bgr in read_video_frames(path, fps):
            rgb = bgr[:, :, ::-1].copy()
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect_for_video(img, int(t * 1000))
            kp = np.full((17, 4), np.nan)
            if res.pose_landmarks:
                lms = res.pose_world_landmarks[0] if self.world else res.pose_landmarks[0]
                norm = res.pose_landmarks[0]
                for c, m in enumerate(MEDIAPIPE_TO_COCO):
                    # world: x direita, y para baixo, z para a câmera → converter para y para cima
                    kp[c] = [lms[m].x, -lms[m].y, -lms[m].z, norm[m].visibility]
            frames.append(kp)
        return np.stack(frames), fps


class YoloPoseEstimator:  # pragma: no cover - exige pesos do YOLOv8
    def __init__(self, weights: str = "yolov8n-pose.pt"):
        from ultralytics import YOLO

        self.model = YOLO(weights)

    def estimate_video(self, path: str | Path, fps: float = 15.0) -> tuple[np.ndarray, float]:
        frames = []
        for _, _, bgr in read_video_frames(path, fps):
            r = self.model(bgr, verbose=False)[0]
            kp = np.full((17, 3), np.nan)
            if r.keypoints is not None and len(r.keypoints.xy):
                h = bgr.shape[0]
                xy = r.keypoints.xy[0].cpu().numpy()
                cf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(17)
                kp[:, 0] = xy[:, 0] / h
                kp[:, 1] = 1 - xy[:, 1] / h  # y para cima, normalizado pela altura
                kp[:, 2] = cf
            frames.append(kp)
        return np.stack(frames), fps


def load_keypoints(path: str | Path) -> tuple[np.ndarray, float]:
    """Carrega keypoints de ``.npy`` ou ``.json`` ({"fps":..., "keypoints": [...]})."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p), 15.0
    d = json.loads(p.read_text(encoding="utf-8"))
    return np.asarray(d["keypoints"], dtype=float), float(d.get("fps", 15.0))


def get_pose_estimator(backend: str = "mediapipe"):
    if backend == "mediapipe":
        return MediaPipePoseEstimator()
    if backend == "yolo":
        return YoloPoseEstimator()
    raise ValueError(backend)
