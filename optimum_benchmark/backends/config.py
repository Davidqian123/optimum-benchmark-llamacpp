import os
from abc import ABC
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, Optional, TypeVar

from psutil import cpu_count

from ..system_utils import get_gpu_device_ids, is_nvidia_system, is_rocm_system

LOGGER = getLogger("backend")


@dataclass
class BackendConfig(ABC):
    name: str
    version: str
    _target_: str

    task: Optional[str] = None
    library: Optional[str] = None
    model_type: Optional[str] = None

    model: Optional[str] = None
    processor: Optional[str] = None

    device: Optional[str] = None
    # we use a string here instead of a list
    # because it's easier to pass in a yaml or from cli
    # and it's consistent with GPU environment variables
    device_ids: Optional[str] = None

    seed: int = 42
    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None

    # model kwargs that are added to its init method/constructor
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    # processor kwargs that are added to its init method/constructor
    processor_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.model is None:
            raise ValueError("`model` must be specified.")

        if self.processor is None:
            self.processor = self.model

        if self.device is None:
            self.device = "cuda" if is_nvidia_system() or is_rocm_system() else "cpu"

        if ":" in self.device:
            LOGGER.warning("`device` was specified using PyTorch format (e.g. `cuda:0`) which is not recommended.")
            self.device = self.device.split(":")[0]
            self.device_ids = self.device.split(":")[1]
            LOGGER.warning(f"`device` and `device_ids` are now set to `{self.device}` and `{self.device_ids}`.")

        if self.device not in ["cuda", "cpu", "mps", "xla", "gpu"]:
            raise ValueError(f"`device` must be either `cuda`, `cpu`, `mps`, `xla` or `gpu`, but got {self.device}")

        if self.device == "cuda":
            if self.device_ids is None:
                LOGGER.warning("`device_ids` was not specified, using all available GPUs.")
                self.device_ids = get_gpu_device_ids()
                LOGGER.warning(f"`device_ids` is now set to `{self.device_ids}` based on system configuration.")

            if is_nvidia_system():
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = self.device_ids
                LOGGER.info(f"CUDA_VISIBLE_DEVICES was set to {os.environ['CUDA_VISIBLE_DEVICES']}.")
            elif is_rocm_system():
                os.environ["ROCR_VISIBLE_DEVICES"] = self.device_ids
                LOGGER.info(f"ROCR_VISIBLE_DEVICES was set to {os.environ['ROCR_VISIBLE_DEVICES']}.")
            else:
                raise RuntimeError("CUDA device is only supported on systems with NVIDIA or ROCm drivers.")

        if self.library not in ["transformers", "diffusers", "timm", "nexa_backend"]:
            raise ValueError(
                f"`library` must be either `transformers`, `diffusers`, `timm` or `nexa_backend`, but got {self.library}"
            )

        if self.inter_op_num_threads is not None:
            if self.inter_op_num_threads == -1:
                self.inter_op_num_threads = cpu_count()

        if self.intra_op_num_threads is not None:
            if self.intra_op_num_threads == -1:
                self.intra_op_num_threads = cpu_count()


BackendConfigT = TypeVar("BackendConfigT", bound=BackendConfig)
