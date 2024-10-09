import importlib.metadata
import importlib.util
from pathlib import Path
from subprocess import STDOUT, check_output
from typing import Optional

_transformers_available = importlib.util.find_spec("transformers") is not None
_accelerate_available = importlib.util.find_spec("accelerate") is not None
_optimum_available = importlib.util.find_spec("optimum") is not None
_torch_available = importlib.util.find_spec("torch") is not None
_onnx_available = importlib.util.find_spec("onnx") is not None
_pynvml_available = importlib.util.find_spec("pynvml") is not None
_torch_distributed_available = importlib.util.find_spec("torch.distributed") is not None
_onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None
_codecarbon_available = importlib.util.find_spec("codecarbon") is not None
_amdsmi_available = importlib.util.find_spec("amdsmi") is not None
_torch_ort_available = importlib.util.find_spec("torch_ort") is not None
_deepspeed_available = importlib.util.find_spec("deepspeed") is not None
_psutil_available = importlib.util.find_spec("psutil") is not None
_optimum_benchmark_available = importlib.util.find_spec("optimum_benchmark") is not None
_pyrsmi_available = importlib.util.find_spec("pyrsmi") is not None
_nexa_sdk_available = importlib.util.find_spec("nexaai") is not None



def is_nexa_sdk_available():
    return _nexa_sdk_available


def is_pyrsmi_available():
    return _pyrsmi_available


def is_psutil_available():
    return _psutil_available


def is_transformers_available():
    return _transformers_available


def is_deepspeed_available():
    return _deepspeed_available


def is_torch_ort_available():
    return _torch_ort_available


def is_accelerate_available():
    return _accelerate_available


def is_onnx_available():
    return _onnx_available


def is_optimum_available():
    return _optimum_available


def is_onnxruntime_available():
    return _onnxruntime_available


def is_pynvml_available():
    return _pynvml_available


def is_amdsmi_available():
    return _amdsmi_available


def is_torch_available():
    return _torch_available


def is_torch_distributed_available():
    return _torch_distributed_available


def is_codecarbon_available():
    return _codecarbon_available


def torch_version():
    if is_torch_available():
        return importlib.metadata.version("torch")


def onnxruntime_version():
    try:
        return "ort:" + importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        try:
            return "ort-gpu:" + importlib.metadata.version("onnxruntime-gpu")
        except importlib.metadata.PackageNotFoundError:
            try:
                return "ort-training:" + importlib.metadata.version("onnxruntime-training")
            except importlib.metadata.PackageNotFoundError:
                return None



def optimum_version():
    if _optimum_available:
        return importlib.metadata.version("optimum")


def transformers_version():
    if _transformers_available:
        return importlib.metadata.version("transformers")


def accelerate_version():
    if _accelerate_available:
        return importlib.metadata.version("accelerate")


def torch_ort_version():
    if _torch_ort_available:
        return importlib.metadata.version("torch_ort")


def optimum_benchmark_version():
    if _optimum_benchmark_available:
        return importlib.metadata.version("optimum_benchmark")


def nexa_sdk_version():
    if _nexa_sdk_available:
        return importlib.metadata.version("nexaai")


def get_git_revision_hash(package_name: str) -> Optional[str]:
    """
    Returns the git commit SHA of a package installed from a git repository.
    """

    try:
        path = Path(importlib.util.find_spec(package_name).origin).parent
    except Exception:
        return None

    try:
        git_hash = check_output(["git", "rev-parse", "HEAD"], cwd=path, stderr=STDOUT).strip().decode("utf-8")

    except Exception:
        return None

    return git_hash


def get_hf_libs_info():
    return {
        "optimum_benchmark_version": optimum_benchmark_version(),
        "optimum_benchmark_commit": get_git_revision_hash("optimum_benchmark"),
        "transformers_version": transformers_version() if is_transformers_available() else None,
        "transformers_commit": get_git_revision_hash("transformers"),
        "accelerate_version": accelerate_version() if is_accelerate_available else None,
        "accelerate_commit": get_git_revision_hash("accelerate"),
        "optimum_version": optimum_version() if is_optimum_available() else None,
        "optimum_commit": get_git_revision_hash("optimum"),
    }
