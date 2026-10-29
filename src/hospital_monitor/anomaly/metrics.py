"""Métricas de avaliação: ponto a ponto e por episódio (com atraso de detecção)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Scores:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _prf(tp: int, fp: int, fn: int) -> Scores:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return Scores(p, r, f, tp, fp, fn)


def pointwise(y_true, y_pred) -> Scores:
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    return _prf(int((y_true & y_pred).sum()), int((~y_true & y_pred).sum()), int((y_true & ~y_pred).sum()))


def runs(mask) -> list[tuple[int, int]]:
    """Intervalos [início, fim] (inclusivo) onde ``mask`` é verdadeiro."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(int)
    d = np.diff(m)
    return [(s, e - 1) for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def event_level(y_true, y_pred, step_minutes: float = 1.0) -> dict:
    """Episódio verdadeiro é detectado se algum episódio previsto o intersecta.

    Retorna precision/recall por episódio e o atraso médio (min) entre o
    início real e a primeira detecção.
    """
    t_runs, p_runs = runs(y_true), runs(y_pred)
    detected, delays = 0, []
    for s, e in t_runs:
        hits = [ps for ps, pe in p_runs if ps <= e and pe >= s]
        if hits:
            detected += 1
            delays.append(max(0, min(hits) - s) * step_minutes)
    fp = sum(1 for ps, pe in p_runs if not any(ps <= e and pe >= s for s, e in t_runs))
    sc = _prf(detected, fp, len(t_runs) - detected)
    out = sc.to_dict()
    out["mean_delay_min"] = round(float(np.mean(delays)), 1) if delays else None
    out["n_true_episodes"] = len(t_runs)
    return out
