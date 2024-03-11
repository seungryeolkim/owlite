r"""Quantization is a powerful technique used to reduce the storage and computational requirements of deep learning
models. However, this reduction in precision can potentially hurt model accuracy. Calibration is a crucial step in
quantization that helps mitigate this accuracy loss.

Calibration involves measuring the distributions of the activations in the model and using this information
to determine the optimal quantization parameters. This process involves:

1. Collecting data: A representative dataset, called the **calibration dataset**, is used to evaluate
the trained floating-point model.

2. Analyzing data: Statistics about the activation or weight distributions are collected.
Understanding how the data is spread across different values within each layer.

3. Selecting quantization parameters: These parameters, such as the quantization step\_size and zero\_point,
are determined using one of several optimization objectives.
The goal is to find the best balance between minimizing quantization error and preserving model accuracy.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from torch.utils.hooks import RemovableHandle

from ..owlite_core.logger import log

if TYPE_CHECKING:
    from ..nn import FakeQuantizer


class Calibrator(ABC):
    """Base calibrator abstract class

    Uses the forward hook to collect the data needed for calibration and update the quantizer's
    step_size and zero_point.

    In **OwLite**, calibrator classes collect the necessary data for calibration based on the data passing through
    the `FakeQuantizer`. This process enables the determination of the `FakeQuantizer`'s `step_size` and `zero_point`.
    Currently, **OwLite** only supports symmetric quantization, so `zero_point` is fixed to 0.

    Attributes:
        hook_handler (`torch.utils.hooks.RemovableHandle`, `optional`): A hook handler.
        quantizer (`FakeQuantizer`): The `FakeQuantizer` to which the calibration will be applied.
    """

    def __init__(self, quantizer: "FakeQuantizer"):
        self.hook_handler: Optional[RemovableHandle] = None
        self.quantizer: FakeQuantizer = quantizer

    def check_calib_ready(self) -> bool:
        """checks that the conditions for calibration are met

        Returns:
            `True`, if all conditions for calibration are met, `False` otherwise.
        """
        if self.quantizer.calibrator is not self:
            log.error("The calibrator does not match the calibrator that the quantizer points")
            return False
        if self.quantizer.is_enabled:
            log.error("The quantizer should be disabled during calibration.")
            return False
        return True

    @abstractmethod
    def prepare(self) -> RemovableHandle:
        """Prepares calibration for the quantizer.

        Set temporal attributes on the quantizer and register a hook on the quantizer.

        Raises:
            ValueError: If the attributions are already set.

        Returns:
            torch.utils.hooks.RemovableHandle: A registered hook handler.
        """

    @abstractmethod
    def update(self) -> None:
        """Calculate step_size and zero_point of quantizer and update them. Then remove the registered hook."""
