import logging
import random
import string
from abc import ABC
from typing import List, Tuple

# TODO: drop torch dependency and use numpy instead
import torch

LOGGER = logging.getLogger("generators")

DEFAULT_NUM_LABELS = 2
DEFAULT_VOCAB_SIZE = 2
DEFAULT_TYPE_VOCAB_SIZE = 2


class TaskGenerator(ABC):
    def __init__(self, shapes, with_labels: bool):
        self.shapes = shapes
        self.with_labels = with_labels

    @staticmethod
    def generate_random_integers(min_value: int, max_value: int, shape: Tuple[int]):
        return torch.randint(min_value, max_value, shape)

    @staticmethod
    def generate_random_floats(min_value: float, max_value: float, shape: Tuple[int]):
        return torch.rand(shape) * (max_value - min_value) + min_value

    @staticmethod
    def generate_ranges(start: int, stop: int, shape: Tuple[int]):
        return torch.arange(start, stop).repeat(shape[0], 1)

    @staticmethod
    def generate_random_strings(num_seq: int) -> List[str]:
        return [
            "".join(random.choice(string.ascii_letters + string.digits) for _ in range(random.randint(10, 100)))
            for _ in range(num_seq)
        ]

    def __call__(self):
        raise NotImplementedError("Generator must implement __call__ method")


class TextGenerator(TaskGenerator):
    def input_ids(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["vocab_size"] or DEFAULT_VOCAB_SIZE,
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def attention_mask(self):
        return self.generate_random_integers(
            min_value=1,  # avoid sparse attention
            max_value=2,
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def token_type_ids(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["type_vocab_size"] or DEFAULT_TYPE_VOCAB_SIZE,
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def position_ids(self):
        return self.generate_ranges(
            start=0,
            stop=self.shapes["sequence_length"],
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def requires_token_type_ids(self):
        return self.shapes["type_vocab_size"] is not None and self.shapes["type_vocab_size"] > 1

    def requires_position_ids(self):
        return self.shapes["max_position_embeddings"] is not None

class TextGenerationGenerator(TextGenerator):
    def __call__(self):
        dummy = {}
        dummy["input_ids"] = self.input_ids()
        dummy["attention_mask"] = self.attention_mask()

        if self.with_labels:
            dummy["labels"] = self.input_ids()

        return dummy


TASKS_TO_GENERATORS = {
    # transformers models tasks
    "text-generation": TextGenerationGenerator,
}
