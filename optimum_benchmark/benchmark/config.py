from dataclasses import dataclass, field
from typing import Any, Dict

from ..system_utils import get_system_info


@dataclass
class BenchmarkConfig:
    name: str

    # BACKEND CONFIGURATION
    backend: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386
    # SCENARIO CONFIGURATION
    scenario: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386
    # LAUNCHER CONFIGURATION
    launcher: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386

    # ENVIRONMENT CONFIGURATION
    environment: Dict[str, Any] = field(default_factory=lambda: {**get_system_info()})

    print_report: bool = False
    log_report: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkConfig":
        return cls(**data)

    @property
    def default_filename(cls) -> str:
        return "benchmark_config.json"
