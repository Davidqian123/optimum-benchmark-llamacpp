from optimum_benchmark import (
    Benchmark,
    BenchmarkConfig,
    InferenceConfig,
    ProcessConfig,
    LlamaCppConfig,
)
from optimum_benchmark.logging_utils import setup_logging

setup_logging(level="INFO", prefix="MAIN-PROCESS")

def run_power_inference(model_path: str):
    BENCHMARK_NAME = f"nexa_sdk_{model_path}"

    launcher_config = ProcessConfig()
    backend_config = LlamaCppConfig(
        device="cuda",
        model=model_path,
        task="text-generation",
    )
    scenario_config = InferenceConfig(
        latency=True,
        memory=True,
        energy=True,
        input_shapes={
            "batch_size": 1,
            "sequence_length": 256,
            "vocab_size": 32000,
        },
        generate_kwargs={
            "max_new_tokens": 100,
            "min_new_tokens": 100,
        },
    )

    # Combine all configurations into the benchmark configuration
    benchmark_config = BenchmarkConfig(
        name=BENCHMARK_NAME,
        launcher=launcher_config,
        backend=backend_config,
        scenario=scenario_config,
    )

    # Launch the benchmark with the specified configuration
    benchmark_report = Benchmark.launch(benchmark_config)

    # Optionally, create a Benchmark object with the config and report
    benchmark = Benchmark(config=benchmark_config, report=benchmark_report)

    # save artifacts to disk as json or csv files
    benchmark_report.save_csv(f"benchmark_report_{model_path}.csv") # or benchmark_report.save_json("benchmark_report.json")

if __name__ == "__main__":
    run_power_inference("/home/ubuntu/.cache/nexa/hub/official/gemma-1.1-2b-instruct/q4_0.gguf")

