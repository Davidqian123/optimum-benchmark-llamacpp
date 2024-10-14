from tempfile import TemporaryDirectory
from typing import Any, ClassVar, Dict
from logging import getLogger
import random
import numpy as np

from nexa.gguf import NexaTextInference
from .config import NexaConfig

class NexaBackend:
    NAME: ClassVar[str] = "nexa_backend"

    def __init__(self, config: NexaConfig) -> None:
        self.config = config

        self.logger = getLogger(self.NAME)
        self.logger.info(f"Allocating {self.NAME}")

        self.logger.info(f"\t+ Seeding backend with {self.config.seed}")
        self.seed()

        self.logger.info("\t+ Benchmarking a nexa model")
        self.pretrained_processor = None
        self.generation_config = None
        self.pretrained_config = None
        self.automodel_loader = None
        # TODO: need a custom method to extract shapes from gguf

    def seed(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def prepare_input_shapes(self, input_shapes: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.task == "text-generation":
            if input_shapes["batch_size"] != 1:
                raise ValueError("Batch size must be 1 for text generation")
        return input_shapes

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"tokens": inputs["input_ids"].squeeze(0).tolist()}

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

    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> list[int]:
        next(self.pretrained_model.generate(**inputs))

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> list[int]:
        generator = self.pretrained_model.generate(**inputs)
        for _ in range(kwargs["max_new_tokens"]):
            next(generator)
