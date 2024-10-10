from dataclasses import dataclass
from typing import Optional

from ...import_utils import nexa_sdk_version
from ..config import BackendConfig


@dataclass
class NexaConfig(BackendConfig):
    name: str = "nexa_backend"
    version: Optional[str] = nexa_sdk_version()
    _target_: str = "optimum_benchmark.backends.nexa_backend.backend.NexaBackend"

    no_weights: bool = False
    filename: Optional[str] = None

    def __post_init__(self):
        self.library = "nexa_backend"
        self.model_type = "nexa_backend"

        super().__post_init__()

        if self.task != "text-generation":
            raise NotImplementedError(f"Task {self.task} is not supported by Nexa backend.")

        if self.no_weights:
            raise NotImplementedError("`no_weights` benchmarking is not supported by Nexa backend.")
