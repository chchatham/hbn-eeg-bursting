from dataclasses import dataclass, field


@dataclass
class MethodResult:
    method_name: str
    detected_bursts: list
    headline_stat: float
    headline_stat_name: str
    metadata: dict = field(default_factory=dict)
