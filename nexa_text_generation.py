from optimum_benchmark import (
    Benchmark,
    BenchmarkConfig,
    InferenceConfig,
    ProcessConfig,
    NexaConfig,
)
from optimum_benchmark.logging_utils import setup_logging
import argparse

setup_logging(level="INFO", prefix="MAIN-PROCESS")

def run_power_inference(model_path: str, device: str, new_tokens: int):
    BENCHMARK_NAME = f"nexa_sdk_{model_path}"

    launcher_config = ProcessConfig()
    backend_config = NexaConfig(
        device=device,
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
            "max_new_tokens": new_tokens,
            "min_new_tokens": new_tokens,
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

    # save artifacts to disk as json or csv files
    benchmark_report.save_csv(f"benchmark_report_{model_path}.csv") # or benchmark_report.save_json("benchmark_report.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run power inference with a specified model"
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path or identifier for the model in Nexa Model Hub",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device to run the inference on, choose from 'cpu', 'cuda', 'mps'",
    )
    parser.add_argument(
        "-n",
        "--new_tokens",
        type=int,
        default=100,
        help="Number of new tokens to generate",
    )
    
    args = parser.parse_args()
    run_power_inference(args.model_path, args.device, args.new_tokens)

