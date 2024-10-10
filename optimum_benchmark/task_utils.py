import importlib
import json
import os
from typing import Optional

import huggingface_hub

from .backends.transformers_utils import (
    TASKS_TO_MODEL_LOADERS,
    get_transformers_pretrained_config,
)
from .backends.transformers_utils import (
    TASKS_TO_MODEL_TYPES_TO_MODEL_CLASSES as TRANSFORMERS_TASKS_TO_MODEL_TYPES_TO_MODEL_CLASSES,
)

_SYNONYM_TASK_MAP = {
    "masked-lm": "fill-mask",
    "causal-lm": "text-generation",
    "default": "feature-extraction",
    "vision2seq-lm": "image-to-text",
    "text-to-speech": "text-to-audio",
    "seq2seq-lm": "text2text-generation",
    "translation": "text2text-generation",
    "summarization": "text2text-generation",
    "mask-generation": "feature-extraction",
    "audio-ctc": "automatic-speech-recognition",
    "sentence-similarity": "feature-extraction",
    "speech2seq-lm": "automatic-speech-recognition",
    "sequence-classification": "text-classification",
    "zero-shot-classification": "text-classification",
}

IMAGE_DIFFUSION_TASKS = [
    "inpainting",
    "text-to-image",
    "image-to-image",
]

TEXT_GENERATION_TASKS = [
    "image-to-text",
    "conversational",
    "text-generation",
    "text2text-generation",
    "automatic-speech-recognition",
]

TEXT_EMBEDDING_TASKS = [
    "feature-extraction",
]


def map_from_synonym(task: str) -> str:
    if task in _SYNONYM_TASK_MAP:
        task = _SYNONYM_TASK_MAP[task]
    return task


def infer_library_from_model_name_or_path(
    model_name_or_path: str, revision: Optional[str] = None, token: Optional[str] = None
) -> str:
    inferred_library_name = None

    # if model_name_or_path is a repo
    if huggingface_hub.repo_exists(model_name_or_path, token=token):
        model_info = huggingface_hub.model_info(model_name_or_path, revision=revision, token=token)
        inferred_library_name = getattr(model_info, "library_name", None)

        if inferred_library_name is None:
            repo_files = huggingface_hub.list_repo_files(model_name_or_path, revision=revision, token=token)
            if "model_index.json" in repo_files:
                inferred_library_name = "diffusers"

        if inferred_library_name is None:
            raise RuntimeError(f"Could not infer library name from repo {model_name_or_path}.")

    # if model_name_or_path is a directory
    elif os.path.isdir(model_name_or_path):
        local_files = os.listdir(model_name_or_path)

        if "model_index.json" in local_files:
            inferred_library_name = "diffusers"
        elif "config.json" in local_files:
            config_dict = json.load(open(os.path.join(model_name_or_path, "config.json"), "r"))
            if "pretrained_cfg" in config_dict or "architecture" in config_dict:
                inferred_library_name = "timm"
            elif "_diffusers_version" in config_dict:
                inferred_library_name = "diffusers"
            else:
                inferred_library_name = "transformers"

        if inferred_library_name is None:
            raise KeyError(f"Could not find the proper library name for directory {model_name_or_path}.")

    else:
        raise KeyError(
            f"Could not find the proper library name for {model_name_or_path}"
            " because it's neither a repo nor a directory."
        )

    return inferred_library_name


def infer_task_from_model_name_or_path(
    model_name_or_path: str,
    library_name: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    if library_name is None:
        library_name = infer_library_from_model_name_or_path(model_name_or_path, revision=revision, token=token)

    inferred_task_name = None

    if os.path.isdir(model_name_or_path):
        if library_name == "transformers":
            transformers_config = get_transformers_pretrained_config(model_name_or_path, revision=revision, token=token)
            auto_modeling_module = importlib.import_module("transformers.models.auto.modeling_auto")
            model_type = transformers_config.model_type

            for task_name, model_loaders in TRANSFORMERS_TASKS_TO_MODEL_TYPES_TO_MODEL_CLASSES.items():
                if isinstance(model_loaders, str):
                    model_loaders = (model_loaders,)
                for model_loader in model_loaders:
                    model_loader_class = getattr(auto_modeling_module, model_loader)
                    model_mapping = model_loader_class._model_mapping._model_mapping
                    if model_type in model_mapping:
                        inferred_task_name = task_name
                        break
                if inferred_task_name is not None:
                    break

    elif huggingface_hub.repo_exists(model_name_or_path, token=token):
        model_info = huggingface_hub.model_info(model_name_or_path, revision=revision, token=token)

        if model_info.pipeline_tag is not None:
            inferred_task_name = map_from_synonym(model_info.pipeline_tag)

        elif inferred_task_name is None:
            if model_info.transformers_info is not None and model_info.transformersInfo.pipeline_tag is not None:
                inferred_task_name = map_from_synonym(model_info.transformersInfo.pipeline_tag)
            else:
                auto_model_class_name = model_info.transformers_info["auto_model"]
                for task_name, model_loaders in TASKS_TO_MODEL_LOADERS.items():
                    if isinstance(model_loaders, str):
                        model_loaders = (model_loaders,)
                    for model_loader in model_loaders:
                        if auto_model_class_name == model_loader:
                            inferred_task_name = task_name
                            break
                    if inferred_task_name is not None:
                        break

    if inferred_task_name is None:
        raise KeyError(f"Could not find the proper task name for {auto_model_class_name}.")

    return inferred_task_name


def infer_model_type_from_model_name_or_path(
    model_name_or_path: str,
    library_name: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> str:
    if library_name is None:
        library_name = infer_library_from_model_name_or_path(model_name_or_path, revision=revision, token=token)

    inferred_model_type = None

    if library_name == "nexa_backend":
        inferred_model_type = "nexa_backend"

    else:
        transformers_config = get_transformers_pretrained_config(
            model_name_or_path, revision=revision, token=token, trust_remote_code=trust_remote_code
        )
        inferred_model_type = transformers_config.model_type

    if inferred_model_type is None:
        raise KeyError(f"Could not find the proper model type for {model_name_or_path}.")

    return inferred_model_type
