# src/analysis/target_spec.py

from dataclasses import dataclass

@dataclass
class TargetSpec:
    name: str
    task_type: str          # "regression" | "classification"
    description: str
    allowed_null_ratio: float = 0.05
