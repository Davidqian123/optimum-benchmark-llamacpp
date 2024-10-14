from dataclasses import asdict, dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Type, Union, Dict, Any

import pandas as pd
from flatten_dict import flatten
from hydra.utils import get_class

from ..backends.config import NexaConfig
from ..launchers import LauncherConfig
from ..scenarios import ScenarioConfig
from .config import BenchmarkConfig
from .report import BenchmarkReport

if TYPE_CHECKING:
    from ..backends.nexa_backend import NexaBackend
    from ..launchers.base import Launcher
    from ..scenarios.base import Scenario


LOGGER = getLogger("benchmark")


@dataclass
class Benchmark:
    config: BenchmarkConfig
    report: BenchmarkReport

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = BenchmarkConfig.from_dict(self.config)
        elif not isinstance(self.config, BenchmarkConfig):
            raise ValueError("config must be either a dict or a BenchmarkConfig instance")

        if isinstance(self.report, dict):
            self.report = BenchmarkReport.from_dict(self.report)
        elif not isinstance(self.report, BenchmarkReport):
            raise ValueError("report must be either a dict or a BenchmarkReport instance")

    @staticmethod
    def launch(config: BenchmarkConfig):
        """
        Runs an benchmark using specified launcher configuration/logic
        """

        # Allocate requested launcher
        launcher_config: LauncherConfig = config.launcher
        launcher_factory: Type[Launcher] = get_class(launcher_config._target_)
        launcher: Launcher = launcher_factory(launcher_config)

        # Launch the benchmark using the launcher
        report = launcher.launch(worker=Benchmark.run, worker_args=[config])

        if config.log_report:
            report.log()

        if config.print_report:
            report.print()

        return report

    @staticmethod
    def run(config: BenchmarkConfig):
        """
        Runs a scenario using specified backend configuration/logic
        """

        # Allocate requested backend
        backend_config: NexaConfig = config.backend
        backend_factory: Type[NexaBackend] = get_class(backend_config._target_)
        backend: NexaBackend = backend_factory(backend_config)

        # Allocate requested scenario
        scenario_config: ScenarioConfig = config.scenario
        scenario_factory: Type[Scenario] = get_class(scenario_config._target_)
        scenario: Scenario = scenario_factory(scenario_config)

        # Run the scenario using the backend
        report = scenario.run(backend)

        return report

    def to_dict(self, flat=False) -> Dict[str, Any]:
        data = asdict(self)

        if flat:
            data = flatten(data, reducer="dot")

        return data

    def to_dataframe(self) -> pd.DataFrame:
        flat_dict_data = self.to_dict(flat=True)
        return pd.DataFrame.from_dict(flat_dict_data, orient="index").T    

    def save_csv(self, path: Union[str, Path]) -> None:
        self.to_dataframe().to_csv(path, index=False)

    @property
    def default_filename(self) -> str:
        return "benchmark.json"
