from tempfile import TemporaryDirectory
from typing import Any, Dict

from nexa.gguf import NexaTextInference

from ..base import Backend
from .config import NexaConfig


class NexaBackend(Backend[NexaConfig]):
    NAME: str = "nexa_backend"

    pretrained_model: NexaTextInference

    def __init__(self, config: NexaConfig) -> None:
        super().__init__(config)

    def load(self) -> None:
        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()
        self.logger.info("\t+ Loading pretrained model")
        self.load_model()
        self.tmpdir.cleanup()

    def load_model(self) -> None:
        """
        Load the model from the given model path (normally GGUF, GGML)
        """
        nexa_model = NexaTextInference(model_path=self.config.model)
        self.pretrained_model = nexa_model.model

    def prepare_input_shapes(self, input_shapes: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.task == "text-generation":
            if input_shapes["batch_size"] != 1:
                raise ValueError("Batch size must be 1 for LlamaCpp text generation")
        return input_shapes

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.task == "text-generation":
            return {"tokens": inputs["input_ids"].squeeze(0).tolist()}
        else:
            raise ValueError(f"Task {self.config.task} not supported by {self.NAME}")

    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> list[int]:
        next(self.pretrained_model.generate(**inputs))

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> list[int]:
        generator = self.pretrained_model.generate(**inputs)
        for _ in range(kwargs["max_new_tokens"]):
            next(generator)
