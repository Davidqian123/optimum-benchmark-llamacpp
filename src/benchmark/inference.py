from dataclasses import dataclass
from logging import getLogger

import statistics
from pandas import DataFrame

from src.input.base import InputGenerator
from src.backend.base import Backend
from src.benchmark.base import Benchmark, BenchmarkConfig

BENCHMARK_NAME = "inference"
LOGGER = getLogger(BENCHMARK_NAME)


@dataclass
class InferenceConfig(BenchmarkConfig):
    name: str = BENCHMARK_NAME

    warmup_runs: int = 5
    benchmark_duration: int = 5


class InferenceBenchmark(Benchmark):
    NAME = BENCHMARK_NAME

    def __init__(self, model: str, task: str, device: str):
        super().__init__(model, task, device)

    def configure(self, config: InferenceConfig):
        self.warmup_runs = config.warmup_runs
        self.benchmark_duration = config.benchmark_duration

    def run(self, backend: Backend, input_generator: InputGenerator) -> None:
        LOGGER.info(f"Generating dummy input")
        dummy_inputs = input_generator.generate()

        LOGGER.info(f"Running inference benchmark")
        self.inference_results = backend.run_inference(
            dummy_inputs, self.warmup_runs, self.benchmark_duration)

    def save_results(self, path: str = '') -> None:
        LOGGER.info('Saving inference results')
        self.inference_results.to_csv(path + 'inference_results.csv')
