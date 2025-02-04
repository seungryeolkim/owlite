from types import TracebackType
from typing import Optional

from torch.nn.parallel import DataParallel, DistributedDataParallel

from .backend.fx.types import GraphModuleOrDataParallel
from .enums import OwLiteStatus
from .nn import FakeQuantizer
from .owlite_core.logger import log


def _prepare_for_calibration(model: GraphModuleOrDataParallel) -> None:
    """Create a calibrator and prepare calibration according to opt.

    Args:
        model(`GraphModuleOrDataParallel`): graph module to calibrate.
    """
    log.info("Preparing for calibration")  # UX
    for _, module in model.named_modules(remove_duplicate=True):
        if isinstance(module, (FakeQuantizer,)):
            module.disable()
            module.calibrator.prepare()
    log.info("All fake quantizers in the model are now ready for calibration")  # UX
    log.info("Calibrating the model")  # UX


def _update_fake_quantizers(model: GraphModuleOrDataParallel) -> None:
    """Calculate step size and zero point using data of calibrator and enabling quantization

    Args:
        model(`GraphModuleOrDataParallel`): model to calibrate.
    """
    log.info("Updating fake quantizers based on collected data")
    for name, module in model.named_modules(remove_duplicate=True):
        if isinstance(module, (FakeQuantizer,)):
            module.calibrator.update()
            if module.step_size.abs().max() <= 0:
                log.error(
                    f"({name}) : The step sizes are all zero. Make sure the data is fed to the quantizer correctly"
                )
                continue
            if module.step_size.min() < 0:
                log.warning(
                    f"({name}) : The step size contains a negative number. Automatically changed to positive",
                    stacklevel=2,
                )
                module.step_size.data = module.step_size.data.abs()
            module.enable()
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model.module.meta["owlite_status"] = OwLiteStatus.CALIBRATED
    else:
        model.meta["owlite_status"] = OwLiteStatus.CALIBRATED
    log.info("Updated fake quantizers. Calibration finished")  # UX


class CalibrationContext:
    """ContextManager for calibration"""

    def __init__(self, model: GraphModuleOrDataParallel):
        self.model = model

    def __enter__(self) -> GraphModuleOrDataParallel:
        _prepare_for_calibration(self.model)
        return self.model

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        _update_fake_quantizers(self.model)


def calibrate(model: GraphModuleOrDataParallel) -> CalibrationContext:
    """Calibration is performed using the supplied data within a 'with' statement.

    `owlite.calibrate` performs Post-Training Quantization (PTQ) calibration on a model converted with the
    `OwLite.convert`. It is required to preserve the model's accuracy by carefully selecting the quantization
    hyperparameters (the scale and zero-point). PTQ calibration typically requires only a subset of the training data.

    Please review the [Calibration document](https://squeezebits.gitbook.io/owlite/python-api/owlite.calibrators)
    in USER GUIDE/Quantization for technical details.

    Args:
        model(`GraphModuleOrDataParallel`): GraphModule or DataParallel model to calibrate.

    Returns:
        CalibrationContext

    ### Usage

    `owlite.calibrate` returns an `owlite.CalibratorContext` object from the OwLite library can be used with a `with`
    statement to perform calibration. The `CalibratorContext` prepares the model for calibration and updates
    the model's fake quantizers after calibration is complete.

    **Example**

    ```python
    with owlite.calibrate(model):
        for i, data in enumerate(train_loader):
            model(*data) # feed data to model and store information from it.
        # calculate fake quantizers step_sizes and zero_points

    # You should use the `model` outside of the block after the calibration
    torch.save(model.state_dict())
    ```

    In this example, the `owlite.calibrate` creates an `owlite.CalibratorContext`,
    referenced by the variable `calibrator`. The training data fetched from `train_loader`
    are then passed to the `calibrator` to perform calibration.

    Note that you should continue writing your code outside of the `with` block since the fake quantizers
    in the model are updated as the `with` block exits.

    """
    return CalibrationContext(model)
