from typing import Any, Dict

from .task_generator import TASKS_TO_GENERATORS, TaskGenerator


class InputGenerator:
    task_generator: TaskGenerator

    def __init__(self, task: str, input_shapes: Dict[str, int]) -> None:
        shapes = {**input_shapes}
        self.task_generator = TASKS_TO_GENERATORS[task](shapes=shapes, with_labels=False)

    def __call__(self) -> Dict[str, Any]:
        task_input = self.task_generator()
        return task_input
