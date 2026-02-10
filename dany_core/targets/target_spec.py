from dataclasses import dataclass


@dataclass
class TargetSpec:
    name: str
    task_type: str  # "classification" or "regression"
    description: str = ""
    allowed_null_ratio: float = 0.0
