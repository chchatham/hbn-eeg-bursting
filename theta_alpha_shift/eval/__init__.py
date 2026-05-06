from dataclasses import dataclass, field


@dataclass
class EvalResult:
    method_name: str
    regime: str
    age: int
    trial: int
    headline_stat: float
    headline_stat_name: str
    metadata: dict = field(default_factory=dict)
