"""This module allows user to train networks in distributed manner using `horovod`

!!!note

    __IMPORTANT__: This module is experimental and may not be working
    correctly. Use at your own risk and report any issues you find.

!!!note

    __IMPORTANT__: This module needs `horovod` Python package to be visible.
    You can install it with `pip install -U torchtraining[horovod]`.
    Also you should export `CUDA_HOME` variable like this:
    `CUDA_HOME=/opt/cuda pip install -U torchtraining[horovod]` (your path may vary)



See `Horovod documentation [https://github.com/horovod/horovod](https://github.com/horovod/horovod)`__ for details
about the framework (installation, capabilities etc.).


Example::

    import torchtraining as tt
    import torchtraining.accumulators.horovod as horovod


    class TrainStep(tt.steps.Train):
        def forward(self, module, sample):
            # Dummy step
            images, labels = sample
            return loss


    model = ...
    criterion = ...
    dataset = ...
    optimizer = ...
    writer = ...


    # Accelerate!
    accelerator = tt.accelerators.Horovod(model, optimize.optimizer)

    # Distributed optimization with gradient accumulation
    optimizer = horovod.optimizer(optimizer, module.named_parameters())

    # Special distributed DataLoader
    dataloader = horovod.DataLoader(dataset, batch_size=64)


    step = (
        TrainStep(criterion, device)
        ** tt.pytorch.ZeroGrad()
        ** tt.pytorch.Backward()
        ** tt.pytorch.Optimize(optimizer)
    )
    iteration = (
        ** tt.iterations.TrainIteration(step, model, dataloader)
        ** horovod.AllReduce()
        ** tt.accumulators.Mean()
        ** horovod.OnRank(tt.callbacks.Tensorboard(writer, "Loss"))
    )

Specific `operations` integrated by `torchtraining` below.

"""

import operator
import pathlib
import pickle
import typing

import torch

import horovod.torch as hvd

from .._base import Operation


def _reduction(name):
    mapping = {
        "sum": hvd.Sum,
        "mean": hvd.Average,
    }
    value = mapping.get(name.lower())
    if value is None:
        raise ValueError(
            "reduction can be one of {}, got {}".format(mapping.keys(), name)
        )
    return value


def _compression(name):
    mapping = {
        "none": hvd.compression.NoneCompressor(),
        "fp16": hvd.compression.FP16Compressor(),
    }
    value = mapping.get(name.lower())
    if value is None:
        raise ValueError(
            "compression can be one of {}, got {}".format(mapping.keys(), compression)
        )
    return value


class OnRank(Operation):
    """Run any operation only if it runs in specified process (specified rank).

    Otherwise return unchanged `data`.

    Attributes:
        operation:
            Operation to run (`callbacks`, `metrics` and whatever else you want).
        rank:
            Rank (process) on which the operation will be run. Default: `0` (main process)

    Returns:
        data | operation(data)
            If run in specified process, return `operation(data)`. Otherwise forward
            data without changes.

    """

    def __init__(
        self, operation: Operation, rank: int = 0,
    ):
        """Initialize `OnRank` object.
    
        Arguments:
            operation:
                Operation to run (`callbacks`, `metrics` and whatever else you want).
            rank:
                Rank (process) on which the operation will be run. Default: `0` (main process)
    """
    Returns:
        data | operation(data)
            If run in specified process, return `operation(data)`. Otherwise forward
            data without changes.
        self.operation = operation
        self.rank = rank

    def forward(self, data: typing.Any):
        """
        Arguments:
            data:
                Input required by `operation`

        """
        if hvd.rank() == self.rank:
            return self.operation(data)
        return data


class DataLoader(torch.utils.data.DataLoader):
    """PyTorch `torch.utils.data.DataLoader` suited for `horovod` integration.

    Works exactly like it's PyTorch counterpart but creates appropriate
    `torch.utils.data.DistributedSampler` under the hood (hence users cannot
    specify `sampler` or `batch_sampler`).

    Attributes:
        dataset:
            Dataset from which to load the data.
        batch_size:
            How many samples per batch to load. Default: ``1``
        shuffle:
            Set to ``True`` to have the data reshuffled at every epoch.
            Default: ``False``
        num_workers:
            How many subprocesses to use for data loading.
            ``0`` means that the data will be loaded in the main process.
            Default: ``0``
        collate_fn Callable, optional
            Merges a list of samples to form a mini-batch of Tensor(s).
            Used when using batched loading from a map-style dataset.
            Default: `None` (default PyTorch collation)
        pin_memory:
            If ``True``, the data loader will copy `torch.Tensors`
            into CUDA pinned memory before returning them. Default: `False`
        drop_last:
            Set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. Default: ``False``
        timeout:
            If positive, the timeout value for collecting a batch
            from workers. Should always be non-negative.
            Default: ``0``
        worker_init_fn:
            If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading.
            Default: ``None``

    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        sampler_seed=0,
    ):
        """Initialize `DataLoader` object.
        
        Arguments:
            dataset:
                Dataset from which to load the data.
            batch_size: 
                How many samples per batch to load. Default: ``1``
            shuffle: 
                Set to ``True`` to have the data reshuffled at every epoch.
                Default: ``False``
            num_workers:
                How many subprocesses to use for data loading.
                ``0`` means that the data will be loaded in the main process.
                Default: ``0``
                collate_fn Callable, optional
                Merges a list of samples to form a mini-batch of Tensor(s).
                Used when using batched loading from a map-style dataset.
                Default: `None` (default PyTorch collation)
            pin_memory:
                If ``True``, the data loader will copy `torch.Tensors`
                into CUDA pinned memory before returning them. Default: `False`
            drop_last:
                Set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If ``False`` and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. Default: ``False``
            timeout: 
                If positive, the timeout value for collecting a batch
                from workers. Should always be non-negative.
                Default: ``0``
            worker_init_fn:
                If not ``None``, this will be called on each
                worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
                input, after seeding and before data loading.
                Default: ``None``
        """
        super().__init__(
            dataset,
            batch_size,
            False,
            torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=hvd.size(),
                rank=hvd.rank(),
                shuffle=shuffle,
                seed=sampler_seed,
            ),
            None,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
        )


class AllReduce(Operation):
    """Perform reduction of the input tensor over all the processes.

    If `data` requires gradient you can backpropagate through this operation.

    Attributes:
        reduction:
            The reduction operation to use when combining gradients across different
            processes. Can be one of: ["mean", "sum"] being respectively:
            [hvd.mpi_ops.Average, hvd.mpi_ops.Sum].
            Default: "mean"
        compression:
            Compression algorithm used during allreduce to reduce the amount of
            data sent during the each parameter update step.
            Can be one of "none" or "fp16". Default: "none"
        name:
            Name of the reduction operator. If not provided it will be generated
            automatically. Default: `None` (automatic generation)

    Returns:
        torch.Tensor:
            Tensor with the same shape as `data` averaged (`reduction="mean"`) or
            summed (`reduction="sum"`) across all processes.

    """

    def __init__(self, reduction: str = "mean", compression="none", name=None):
        """Initialize `AllReduce` object.
        
        Arguments:
            reduction:
                The reduction operation to use when combining gradients across different
                processes. Can be one of: ["mean", "sum"] being respectively:
                [hvd.mpi_ops.Average, hvd.mpi_ops.Sum].
                Default: "mean"
            compression:
                Compression algorithm used during allreduce to reduce the amount of
                data sent during the each parameter update step.
                Can be one of "none" or "fp16". Default: "none"
            name:
                Name of the reduction operator. If not provided it will be generated
                automatically. Default: `None` (automatic generation)
        """
        self.name = name
        self.reduction = _reduction(reduction)
        self.compression = _compression(compression)

    def forward(self, data):
        """
        Arguments:  
            data:
                Tensor to be reduced
        """
        return hvd.allreduce(
            data, name=self.name, compression=self.compression, op=self.reduction
        )


class AsyncAllReduce(Operation):
    """Perform asynchronous reduction of the input tensor over all the processes.

    User should pipe this object into `tt.accelerators.horovod.Synchronize()`
    in order to get value.

    Attributes:
        reduction:
            The reduction operation to use when combining gradients across different
            processes. Can be one of: ["mean", "sum"] being respectively:
            [hvd.mpi_ops.Average, hvd.mpi_ops.Sum].
            Default: "mean"
        name:
            Name of the reduction operator. If not provided it will be generated
            automatically. Default: `None` (automatic generation)


    Returns:
        Handle
            Handle to be used with `tt.accelerators.horovod.Synchronize()`

    """

    def __init__(self, reduction: str = "mean", compression="none", name=None):
        """Initialize `AsyncAllReduce` object.
        
        Arguments:
            reduction:
                The reduction operation to use when combining gradients across different
                processes. Can be one of: ["mean", "sum"] being respectively:
                [hvd.mpi_ops.Average, hvd.mpi_ops.Sum].
                Default: "mean"
            compression:
                Compression algorithm used during allreduce to reduce the amount of
                data sent during the each parameter update step.
                Can be one of "none" or "fp16". Default: "none"
            name:
                Name of the reduction operator. If not provided it will be generated
                automatically. Default: `None` (automatic generation)
        """
        self.name = name
        self.reduction = _reduction(reduction)

    def forward(self, data):
        """
        Arguments:
            data:
                Tensor to be reduced across all processes.
        """
        return hvd.allreduce_async(data, name=self.name, op=self.reduction)


class AllGather(Operation):
    """Concatenate input tensors from all processes.

    Tensor after concatenation will be available to all processes.
    Concatenation is done over `0` th dimension, so it's the only dimension
    in which `torch.Tensor` on different processes is allowed to be different.

    If `data` requires gradient you can backpropagate through this operation.

    Attributes:
        name:
            Name of the reduction operator. If not provided it will be generated
            automatically. Default: `None` (automatic generation)

    Returns:
        torch.Tensor:
            Tensor with the same shape as `data` except `0` dimension (which will be larger
            as it's concatenation of data from all processes).

    """

    def __init__(self, name: str = None):
        """Initialize `AllGather` object.
        
        Arguments:
            name:
                Name of the reduction operator. If not provided it will be generated
                automatically. Default: `None` (automatic generation)
        """
        self.name = name

    def forward(self, data):
        """
        Arguments:
            data:
                Tensor to be gathered across all processes.
        """
        return hvd.allgather(data, name=self.name)


class AsyncAllGather(Operation):
    """Asynchronously concatenate input tensors from all processes.

    Tensor after concatenation will be available to all processes.
    Concatenation is done over `0`th dimension, so it's the only dimension
    in which `torch.Tensor` on different processes is allowed to be different.

    Attributes:
        name:
            Name of the reduction operator. If not provided it will be generated
            automatically. Default: `None` (automatic generation)

    Returns:
        Handle
            Handle to be used with `tt.accelerators.horovod.Synchronize()`

    """

    def __init__(self, name=None):
        """Initialize `AsyncAllGather` object.
    
        Arguments:
            name:
                Name of the reduction operator. If not provided it will be generated
                automatically. Default: `None` (automatic generation)
        """
        self.name = name

    def forward(self, data):
        """
        Arguments:
            data:
                Tensor to be gathered across all processes.
        """
        return hvd.allgather_async(data, name=self.name,)


class Broadcast(Operation):
    """Broadcast tensor from `rank` process to all other processes.

    If `data` requires gradient you can backpropagate through this operation.

    Attributes:
        rank: 
            Rank of the process from which `data` will be distributed to other processes.
        name: 
            Name of the reduction operator. If not provided it will be generated
            automatically. Default: `None` (automatic generation)

    Returns:
        torch.Tensor:
            Tensor with the same shape as `data` with broadcasted values.

    """

    def __init__(self, rank: int = 0, name=None):
    
        """Initialize `Broadcast` object.
    
        Arguments:
            rank: 
                Rank of the process from which `data` will be distributed to other processes.
            name: 
                Name of the reduction operator. If not provided it will be generated
                automatically. Default: `None` (automatic generation)
    """
        self.rank = rank
        self.name = name

    def forward(self, data):
        """
        Arguments:
            data:
                Tensor to be broadcasted across all processes.
        """
        return hvd.broadcast(data, self.rank, name=self.name)


class AsyncBroadcast(Operation):
    """Asynchronously broadcast tensor from `rank` process to all other processes.

    Attributes:
        rank:
            Rank of the process from which `data` will be distributed to other processes.
        name:
            Name of the reduction operator. If not provided it will be generated
            automatically. Default: `None` (automatic generation)

    Returns:
        Handle
            Handle to be used with `tt.accelerators.horovod.Synchronize()`

    """

    def __init__(self, rank: int = 0, name=None):
        """Initialize `AsyncBroadcast` object.
        
        Arguments:
            rank:
                Rank of the process from which `data` will be distributed to other processes.
            name:
                Name of the reduction operator. If not provided it will be generated
                automatically. Default: `None` (automatic generation)
        """
        self.rank = rank
        self.name = name

    def forward(self, data):
        """
        Arguments:
            data:
                Tensor to be broadcasted across all processes.
        """
        return hvd.async_broadcast(data, self.rank, name=self.name)


class Synchronize(Operation):
    """Asynchronously broadcast tensor from `rank` process to all other processes.

    Returns:
        torch.Tensor:
            Value of the previous asynchronous operation after synchronization.
            Whatever it should return.

    """

    def forward(self, handle):
        """
        Arguments:
            handle:
                Handle returned by an `AsyncAllReduce`, `AsyncAllGather`or
                `AsyncBroadcast` which will be used to retrieve `torch.Tensor`.
        """
        return hvd.synchronize(handle)


def optimizer(
    optimizer,
    named_parameters,
    reduction: str = "sum",
    compression: str = "none",
    accumulate: int = 1,
    rank: int = 0,
):
    """Create Horovod compatible optimizer.

    State of optimizer will be distributed on specified `rank`.
    Should be used after `torchtraining.accelerators.Horovod` object was created.

    Arguments:
        optimizer:
            Instance of optimizer-like object with interface aligned with
            `torch.optim.Optimizer`.
        named_parameters:
            A mapping between parameter names and values. Used for naming of allreduce operations.
            Typically just `model.named_parameters()`.
        reduction: str, optional
            The reduction operation to use when combining gradients across different
            processes. Can be one of: ["mean", "sum"] being respectively:
            [hvd.mpi_ops.Average, hvd.mpi_ops.Sum].
            Default: "mean"
        compression:
            Compression algorithm used during allreduce to reduce the amount of
            data sent during the each parameter update step.
            Can be one of "none" or "fp16". Default: "none"
        accumulate:
            Divide loss by ``accumulate`` if gradient accumulation is used.
            This approach averages gradient from multiple batches.
            Default: `1` (no accumulation)
        rank:
            Rank from which optimizer's state will be broadcasted.
            Default: `0`

    Returns:
        horovod.torch.DistributedOptimizer
            Instance of optimizer but distributed across workers.

    """

    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters,
        _compression(compression),
        backward_passes_per_step=accumulate,
        op=_reduction(reduction),
    )

    hvd.broadcast_optimizer_state(optimizer, root_rank=rank)
    return optimizer


def load(f, rank: int = 0, map_location=None, pickle_module=pickle, **pickle_load_args):
    """Load object saved with `torch.save` in a single process and distribute to all other processes.

    Useful when loading saved `torch.nn.Module` (or other `torch` objects like `optimizer`),
    which is saved on a single machine.

    It can be easily distributed to other processes this way.

    If you wish to `torch.save` on a single process you can create an object
    like this::

        save = tt.accelerators.horovod.OnRank(torch.save)
        save(your_module)

    Arguments:
        f: 
            A file-like object (has to implement :meth:`read`, :meth`readline`, :meth`tell`, and :meth`seek`)
            or a string or `os.PathLike` object containing a file name.
        rank: 
            Process rank on which data will be loaded.
        map_location:
            Specifies how to remap storage locations. Default: `None`
        pickle_module:
            Module used for unpickling metadata and objects,
            (has to match the :attr:`pickle_module` used to serialize file).
            Default:
        **pickle_load_args
            optional keyword arguments passed over to :func:`pickle_module.load`
            and :func:`pickle_module.Unpickler`, e.g., :attr:`errors=...`.

    Returns:
        torch.Tensor:
        Anything you saved with `torch.save` really

    """
    data = None
    if hvd.rank() == rank:
        data = torch.load(f, map_location, pickle_module, **pickle_load_args)
    return hvd.broadcast(data, rank)
