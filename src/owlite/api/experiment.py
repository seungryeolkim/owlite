# pylint: disable=duplicate-code
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import onnx
import requests
from packaging.version import Version
from torch.fx.graph_module import GraphModule
from typing_extensions import Self

from ..backend.signature import DynamicSignature, Signature
from ..options import CompressionOptions, DynamicAxisOptions
from ..owlite_core.api_base import DOVE_API_BASE, MAIN_API_BASE
from ..owlite_core.constants import FX_CONFIGURATION_FORMAT_VERSION, OWLITE_VERSION
from ..owlite_core.logger import log
from .baseline import Baseline
from .benchmarkable import Benchmarkable
from .project import Project
from .utils import upload_file_to_url


@dataclass
class Experiment(Benchmarkable):
    """The OwLite experiment"""

    baseline: Baseline
    has_config: bool
    input_signature: Optional[Union[Signature, DynamicSignature]] = field(default=None)

    @property
    def project(self) -> Project:
        """The parent project for this experiment"""
        return self.baseline.project

    @property
    def url(self) -> str:
        # TODO (huijong): make this url point to the insight page comparing the experiment against its baseline.
        return self.project.url

    @property
    def home(self) -> str:
        return os.path.join(self.baseline.home, self.name)

    @property
    def label(self) -> str:
        return "_".join((self.project.name, self.baseline.name, self.name))

    @property
    def config(self) -> CompressionOptions:
        """The configuration for this experiment"""
        try:
            resp = DOVE_API_BASE.post("/compile", json=self.payload(format_version=FX_CONFIGURATION_FORMAT_VERSION))
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                log.error(
                    "Config settings exceed the limit. The free plan supports max. 1 OpType, 10 Layer settings. "
                    "Please check your config again"
                )  # UX
            elif e.response.status_code == 426:
                log.error(
                    f"Your current version ({Version(OWLITE_VERSION)}) is not supported. "
                    "Please update the package to the latest version with the following command: "
                    "pip install git+https://github.com/SqueezeBits/owlite --upgrade "
                    "--extra-index-url https://pypi.ngc.nvidia.com"
                )  # UX
            raise e
        assert isinstance(resp, dict)
        return CompressionOptions.load(resp)

    @classmethod
    def create(cls, baseline: Baseline, name: str) -> Self:
        """Creates a new experiment for the baseline.

        Args:
            baseline (Baseline): A baseline
            name (str): The name of the experiment to be created

        Returns:
            Experiment: The newly created experiment
        """
        try:
            _ = MAIN_API_BASE.post(
                "/projects/runs",
                json=baseline.payload(run_name=name),
            )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                err_msg = e.response.json()
                if "is invalid" in err_msg["detail"]:
                    log.error(
                        "Baseline model is invalid. "
                        "You can only create an experiment with an uploaded baseline model. "
                        "Please check if OwLite successfully uploaded the baseline model. "
                        "If not, try using `owl.export(model).`"
                    )  # UX
            raise e
        experiment = cls(name=name, baseline=baseline, has_config=False)
        log.info(f"Created a new {experiment}")  # UX
        baseline.experiments[name] = experiment
        return experiment

    @classmethod
    def load(cls, baseline: Baseline, name: str, *, verbose: bool = True) -> Optional[Self]:
        """Loads the existing experiment named `name` for the given `baseline`

        Args:
            baseline (Baseline): The baseline holding the experiment
            name (str): The name of the experiment to load
            verbose (bool, optional): If True, prints error message when the experiment is not found. Defaults to True.

        Raises:
            e (requests.exceptions.HTTPError): When an unexpected HTTP status code is returned.

        Returns:
            Optional[Experiment]: The loaded experiment if it is found, `None` otherwise.
        """
        try:
            res = MAIN_API_BASE.post("/projects/runs/info", json=baseline.payload(run_name=name))
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                if verbose:
                    log.error(
                        f"No such experiment: {name}. Please check if the experiment name is correct "
                        f"or create a new one at {baseline.project.url}"
                    )  # UX
                return None
            raise e

        assert isinstance(res, dict)
        experiment = cls(name=name, baseline=baseline, has_config=bool(res.get("config_id", "")))
        log.info(f"Loaded the existing {experiment}")  # UX
        baseline.experiments[name] = experiment
        return experiment

    @classmethod
    def load_or_create(cls, baseline: Baseline, name: str) -> Self:
        """Loads the experiment named `name` for the given `baseline` if it already exists, creates a new one otherwise.

        Args:
            baseline (Baseline): The baseline holding the experiment.
            name (str): The name of the experiment to be loaded or created.

        Returns:
            Experiment: The loaded or newly created experiment.
        """
        experiment = cls.load(baseline, name, verbose=False) or cls.create(baseline, name)

        if experiment.has_config:
            log.info(f"Compression configuration found for '{experiment.name}'")  # UX
        else:
            log.warning(f"No compression configuration found for '{experiment.name}'")  # UX

        return experiment

    def clone(self, name: str) -> Self:
        """Clones this experiment.

        Args:
            name (str): The name of the new experiment.

        Raises:
            e (requests.exceptions.HTTPError): When an unexpected HTTP status code is returned.
            RuntimeError: When the experiment to duplicate does not have compression configuration.

        Returns:
            Experiment: The cloned experiment.
        """
        try:
            resp = MAIN_API_BASE.post("/projects/runs/copy", json=self.payload(new_run_name=name))
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                log.error(
                    f"Cannot duplicate experiment. Experiment '{self.name}' doesn't have compression configuration"
                )  # UX
                raise RuntimeError("Compression configuration not found") from e
            raise e
        assert isinstance(resp, dict)
        cloned_experiment = type(self)(name=resp["name"], baseline=self.baseline, has_config=self.has_config)
        log.info(
            f"Copied compression configuration from the {self} to the new experiment '{cloned_experiment.name}'"
        )  # UX
        return cloned_experiment

    def upload(
        self,
        proto: onnx.ModelProto,
        model: Optional[GraphModule] = None,
        dynamic_axis_options: Optional[DynamicAxisOptions] = None,
    ) -> None:
        self.input_signature = Signature.from_onnx(proto, dynamic_axis_options)
        log.debug(f"Experiment signature: {self.input_signature}")

        file_dest_url = MAIN_API_BASE.post(
            "/projects/runs/data/upload",
            json=self.payload(input_shape=json.dumps(self.input_signature)),
        )
        assert file_dest_url is not None and isinstance(file_dest_url, str)
        upload_file_to_url(self.onnx_path, file_dest_url)

    def payload(self, **kwargs: Any) -> dict[str, str]:
        p = {
            "project_id": self.project.id,
            "baseline_name": self.baseline.name,
            "run_name": self.name,
        }
        p.update(kwargs)
        return p

    def __repr__(self) -> str:
        return f"experiment '{self.name}' for the {self.baseline}"
