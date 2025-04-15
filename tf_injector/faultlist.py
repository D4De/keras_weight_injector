from dataclasses import dataclass, field

FaultType = tuple[str, tuple[int, ...], int]

@dataclass
class FaultList:
    # [("layer", (coords,..), bitpos), ...]
    faults: list[FaultType] = field(default_factory=lambda: [])
    resume_idx: int = 0
