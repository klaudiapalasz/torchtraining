"""Cast objects (usually `torch.Tensor`) to specific device.

PyTorch's defaults (as of `1.6.0` CPU and GPU) are forwarded for easier usage.

If you wish to use other (non-standard) devices (like TPUs), please use
`Device` class explicitly, for example (TPU)::


    import torch_xla.core.xla_model as xm

    class TrainStep(tt.steps.Train):
        def forward(self, module, sample):
            # Generate loss and other necessary items
            ...
            return loss, predictions, targets


    step = TrainStep(criterion, device)
    # Select `loss` and perform backpropagation
    step ** tt.Select(loss=0) ** tt.device.Device(xm.xla_device())

!!!note

    __IMPORTANT__: Usually users should use `torchtraining.device.CPU()` cast
    after `iteration` and before `accumulation` in order not to pollute
    GPUs memory.


"""

import importlib

import torch

from ._base import Operation


class CPU(Operation):
    """Cast `object` (usually `torch.Tensor`) to `cpu`.

    !!!note

        __IMPORTANT__: This object should be used most often from this package
        in order to save `GPU`/`TPU` memory.
        See example below

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                return loss, accuracy


        step = TrainStep(criterion, device)
        iteration = tt.iterations.Train(step, module, dataloader)

        # Cast to CPU in order not to inflate GPU memory with `list`
        # You should usually use this OP before accumulation
        iteration ** tt.Select(
            accuracy=1
        ) ** tt.device.CPU() ** tt.accumulators.List() ** tt.callbacks.Logger("Accuracy")

    Attributes:
        memory_format:
            The desired memory format of returned Tensor. Default: torch.preserve_format.
            Default: `torch.preserve_format`

    """

    def __init__(self, memory_format=torch.preserve_format):
        """Initialize `CPU` object.
        
        Arguments:
            memory_format: 
                The desired memory format of returned Tensor. Default: torch.preserve_format.
                Default: `torch.preserve_format`
        """
        super().__init__()
        self.memory_format = memory_format

    def forward(self, data):
        return data.cpu(memory_format=self.memory_format)


class CUDA(Operation):
    """Cast `object` (usually `torch.Tensor`) to cuda enabled device.

    !!!note

        __IMPORTANT__: This object __USUALLY SHOULDN'T BE USED__ as it
        __usually__ pointlessly pollutes GPU memory.


    Attributes:
        device: 
            Device index to select. It’s a no-op if this argument is a negative integer or None.
            Default: `None`
        non_blocking:
            If True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
            Default: `False`
        memory_format: torch.memory_format, optional
            The desired memory format of returned Tensor. Default: torch.preserve_format.
            Default: `torch.preserve_format`

    """

    def __init__(
        self, device=None, non_blocking=False, memory_format=torch.preserve_format
    ):
        """Initialize `CUDA` object.
    
        Arguments:
            device: 
                Device index to select. It’s a no-op if this argument is a negative integer or None.
                Default: `None`
            non_blocking:
                If True and this copy is between CPU and GPU, the copy may occur asynchronously
                with respect to the host. For other cases, this argument has no effect.
                Default: `False`
            memory_format: torch.memory_format, optional
                The desired memory format of returned Tensor. Default: torch.preserve_format.
                Default: `torch.preserve_format`
            
        """
        super().__init__()
        self.device = device
        self.non_blocking = non_blocking
        self.memory_format = memory_format

    def forward(self, data):
        return data.cuda(
            self.device, self.non_blocking, memory_format=self.memory_format,
        )


class Device(Operation):
    """Cast `object` to any device (for example `TPU` with `torch_xla` package).

    !!!note

        __IMPORTANT__: This object __USUALLY SHOULDN'T BE USED__ as it
        __usually__ pointlessly pollutes device memory (unless it's CPU,
        in such case simply use `torchtraining.device.CPU()`).

    See `example` at the beginning of this section.

    Attributes:
        device: 
            Anything which can be used with `torch.Tensor.to` to cast onto
            specified device.

    """

    def __init__(self, device):
        """Initialize ` Device` object.
        
        Arguments:
            device: 
                Anything which can be used with `torch.Tensor.to` to cast onto
                specified device.
        """        
        super().__init__()
        self.device = device

    def forward(self, data):
        return data.to(self.device)
