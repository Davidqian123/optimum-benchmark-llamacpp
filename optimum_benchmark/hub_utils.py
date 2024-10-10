import time
from dataclasses import asdict, dataclass
from json import dump, load
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union

import pandas as pd
from flatten_dict import flatten, unflatten
from huggingface_hub import create_repo, hf_hub_download, upload_file
from huggingface_hub.utils import HfHubHTTPError
from typing_extensions import Self

LOGGER = getLogger("hub_utils")


class classproperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, owner):
        return self.fget(owner)


@dataclass
class PushToHubMixin:
    """
    A Mixin to push artifacts to the Hugging Face Hub
    """

    # DICTIONARY/JSON API
    def to_dict(self, flat=False) -> Dict[str, Any]:
        data = asdict(self)

        if flat:
            data = flatten(data, reducer="dot")

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PushToHubMixin":
        return cls(**data)

    def save_json(self, path: Union[str, Path], flat: bool = False) -> None:
        with open(path, "w") as f:
            dump(self.to_dict(flat=flat), f, indent=4)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> Self:
        with open(path, "r") as f:
            data = load(f)
        return cls.from_dict(data)

    # DATAFRAME/CSV API
    def to_dataframe(self) -> pd.DataFrame:
        flat_dict_data = self.to_dict(flat=True)
        return pd.DataFrame.from_dict(flat_dict_data, orient="index").T

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        data = df.to_dict(orient="records")[0]

        for k, v in data.items():
            if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
                # we correct lists that were converted to strings
                data[k] = eval(v)

            if v != v:
                # we correct nan to None
                data[k] = None

        data = unflatten(data, splitter="dot")
        return cls.from_dict(data)

    def save_csv(self, path: Union[str, Path]) -> None:
        self.to_dataframe().to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> Self:
        return cls.from_dataframe(pd.read_csv(path))

    @classproperty
    def default_filename(self) -> str:
        return "file.json"

    @classproperty
    def default_subfolder(self) -> str:
        return "benchmarks"
