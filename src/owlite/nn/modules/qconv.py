from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.modules.conv import Conv1d, Conv2d, Conv3d, _ConvNd

from ...options import Channel, FakeQuantizerOptions
from .fake_quantizer import FakeQuantizer
from .qmodule_mixins import UnaryNeuralQModuleMixin


# mypy: disable-error-code=misc
class _QConvNd(_ConvNd, UnaryNeuralQModuleMixin):
    """Base class of quantized convolution layer inherited from [torch.nn.modules.conv._ConvNd]
    (https://github.com/pytorch/pytorch/blob/4c55dc50355d5e923642c59ad2a23d6ad54711e7/torch/nn/modules/conv.py).
    It performs convolution operations using the input and fake-quantized weights. Its weights and biases are copied
    from the original convolution instance.

    ## Attributes
    - __input_quantizer__ (`FakeQuantizer`): fake quantizer used for the input.
    - __weight_quantizer__ (`FakeQuantizer`): fake quantizer used for the weights.
    # `class owlite.nn.Qconv1d`
    Quantized 1D convolution module
    # `class owlite.nn.Qconv2d`
    Quantized 2D convolution module
    # `class owlite.nn.Qconv3d`
    Quantized 3D convolution module

    Args:
        _ConvNd (_type_): _description_
        UnaryNeuralQModuleMixin (_type_): _description_
    """

    def __init__(
        self,
        conv: _ConvNd,
        weight_options: Optional[FakeQuantizerOptions] = None,
    ):
        # pylint: disable=no-value-for-parameter
        super().__init__(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,  # type: ignore[arg-type]
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )  # type: ignore
        self.train(conv.training)
        self.input_quantizer: Optional[FakeQuantizer] = None
        channel = Channel(0, self.out_channels) if (weight_options is not None and weight_options.per_channel) else None
        self.weight_quantizer = FakeQuantizer.create(weight_options, channel, narrow_range=True)
        if self.weight_quantizer is not None:
            self.weight_quantizer.to(self.weight.device)
        with torch.no_grad():
            self.weight.copy_(conv.weight)
            if self.bias is not None and conv.bias is not None:
                self.bias.copy_(conv.bias)

    def _set_bias_to_zero(self) -> None:
        self.bias = Parameter(torch.zeros(self.out_channels).to(self.weight.device))

    def forward(self, inputs: Tensor) -> Tensor:
        """forward with quantized weight if available"""
        quant_weight = self.weight_quantizer(self.weight) if self.weight_quantizer is not None else self.weight
        return self._conv_forward(inputs, quant_weight, self.bias)


class QConv1d(_QConvNd, Conv1d):
    """Applies a 1D convolution over an input signal composed of several input planes with quantized weight."""

    def __init__(self, conv: Conv1d, weight_options: Optional[FakeQuantizerOptions] = None):
        """Initializes instance from an existing `torch.nn.Conv1d` instance, copying the weights and bias if it exists.

        Args:
            conv (torch.nn.Conv1d): The original `Conv1d` module to replace with `QConv1d`
            weight_options (Optional[FakeQuantizerOptions], optional): Option for the fake weight quantizer.
                Defaults to None.
        """
        super().__init__(conv, weight_options)


class QConv2d(_QConvNd, Conv2d):
    """Applies a 2D convolution over an input signal composed of several input planes with quantized weight."""

    def __init__(self, conv: Conv2d, weight_options: Optional[FakeQuantizerOptions] = None):
        """Initializes instance from an existing `torch.nn.Conv2d` instance, copying the weights and bias if it exists.

        Args:
            conv (torch.nn.Conv2d): The original `Conv2d` module to replace with `QConv2d`
            weight_options (Optional[FakeQuantizerOptions], optional): Option for the fake weight quantizer.
                Defaults to None.
        """
        super().__init__(conv, weight_options)


class QConv3d(_QConvNd, Conv3d):
    """Applies a 3D convolution over an input signal composed of several input planes with quantized weight."""

    def __init__(self, conv: Conv3d, weight_options: Optional[FakeQuantizerOptions] = None):
        """Initializes instance from an existing `torch.nn.Conv3d` instance, copying the weights and bias if it exists.

        Args:
            conv (torch.nn.Conv3d): The original `Conv3d` module to replace with `QConv3d`
            weight_options (Optional[FakeQuantizerOptions], optional): Option for the fake weight quantizer.
                Defaults to None.
        """
        super().__init__(conv, weight_options)
