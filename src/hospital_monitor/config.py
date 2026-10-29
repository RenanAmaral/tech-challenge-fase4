"""Configuração central: limiares clínicos, pesos de fusão e parâmetros de alerta.

Os valores padrão podem ser sobrescritos por ``config/settings.yaml``. Os
limiares de sinais vitais são inspirados no NEWS2 (Royal College of Physicians)
e servem apenas para fins acadêmicos — não substituem protocolo clínico.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class VitalLimits:
    hr_low: float = 40.0
    hr_high: float = 120.0
    hr_critical_high: float = 140.0
    spo2_low: float = 92.0
    spo2_critical: float = 88.0
    sbp_low: float = 90.0
    sbp_high: float = 180.0
    rr_low: float = 8.0
    rr_high: float = 24.0
    temp_high: float = 38.5


@dataclass
class VitalsDetectorConfig:
    window: int = 60  # amostras (1/min → 1 h) para o z-score móvel
    z_threshold: float = 4.0
    iforest_contamination: float = 0.02
    min_episode_samples: int = 2  # amostras consecutivas para confirmar episódio
    limits: VitalLimits = field(default_factory=VitalLimits)


@dataclass
class FusionConfig:
    window_minutes: int = 15
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "vitals": 1.0,
            "audio": 0.7,
            "text": 0.8,
            "video": 0.7,
            "prescription": 0.9,
            "movement": 0.8,
        }
    )
    agreement_boost: float = 0.15  # por modalidade adicional concordante
    sentinel_kinds: list[str] = field(default_factory=lambda: ["queda", "gauze_count_mismatch"])
    attention_threshold: float = 0.5
    critical_threshold: float = 0.8


@dataclass
class AlertConfig:
    cooldown_minutes: int = 30  # não repetir alerta do mesmo nível no mesmo paciente
    min_level: str = "attention"


@dataclass
class Settings:
    vitals: VitalsDetectorConfig = field(default_factory=VitalsDetectorConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    outputs_dir: Path = ROOT / "outputs"
    data_dir: Path = ROOT / "data"


def _merge(obj: Any, overrides: dict[str, Any]) -> Any:
    for f in fields(obj):
        if f.name not in overrides:
            continue
        cur = getattr(obj, f.name)
        val = overrides[f.name]
        if is_dataclass(cur) and isinstance(val, dict):
            _merge(cur, val)
        elif isinstance(cur, Path):
            setattr(obj, f.name, Path(val))
        else:
            setattr(obj, f.name, val)
    return obj


def load_settings(path: str | Path | None = None) -> Settings:
    settings = Settings()
    path = Path(path) if path else ROOT / "config" / "settings.yaml"
    if path.exists():
        import yaml

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _merge(settings, data)
    return settings
