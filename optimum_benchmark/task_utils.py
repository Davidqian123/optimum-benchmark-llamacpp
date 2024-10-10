from .backends.transformers_utils import (
    TASKS_TO_MODEL_TYPES_TO_MODEL_CLASSES as TRANSFORMERS_TASKS_TO_MODEL_TYPES_TO_MODEL_CLASSES,
)

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