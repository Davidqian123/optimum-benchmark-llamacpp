from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from logging import getLogger
import os

from psutil import cpu_count

from ..import_utils import nexa_sdk_version
from ..system_utils import get_gpu_device_ids, is_nvidia_system, is_rocm_system

LOGGER = getLogger("backend")

@dataclass
class NexaConfig:
    name: str = "nexa_backend"
    version: Optional[str] = nexa_sdk_version()
    _target_: str = "optimum_benchmark.backends.nexa_backend.NexaBackend"

    # Attributes from BackendConfig
    task: Optional[str] = None
    library: Optional[str] = "nexa_backend"
    model_type: Optional[str] = "nexa_backend"

    model: Optional[str] = None
    processor: Optional[str] = None

    device: Optional[str] = None
    device_ids: Optional[str] = None

    seed: int = 42
    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None

    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    processor_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Specific attribute for NexaConfig
    filename: Optional[str] = None

    def __post_init__(self):
        # Begin code from BackendConfig.__post_init__
        if self.model is None:
            raise ValueError("`model` must be specified.")

        if self.processor is None:
            self.processor = self.model

        if self.device is None:
            if is_nvidia_system() or is_rocm_system():
                self.device = "cuda"
            else:
                self.device = "cpu"

        if self.device not in ["cuda", "cpu", "mps"]:
            raise ValueError(f"`device` must be either `cuda`, `cpu`, or `mps`, but got {self.device}")

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
            
        if self.inter_op_num_threads is not None:
            if self.inter_op_num_threads == -1:
                self.inter_op_num_threads = cpu_count()

        if self.intra_op_num_threads is not None:
            if self.intra_op_num_threads == -1:
                self.intra_op_num_threads = cpu_count()


