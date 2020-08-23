"""Traditionally known callback-like pipes.

This module allows user to `save` their best model, terminate training
earlier or Log `data` into `loguru.logger`.

Example::

    class TrainStep(tt.steps.Train):
        def forward(self, module, sample):
            ...
            return loss, targets


    step = TrainStep(criterion, device)
    step > tt.Select(loss=0) > tt.callbacks.TerminateOnNan()

Users can also use specific `callbacks` which integrate with third party tools,
namely:

    * tensorboard
    * neptune
    * comet

"""

import importlib
import numbers
import operator
import pathlib
import sys
import time
import typing

import loguru
import torch

from .. import _base, exceptions
from ..utils import general as utils

# Conditional integration imports

if utils.modules_exist("torch.utils.tensorboard"):
    from . import tensorboard


if utils.modules_exist("neptune"):
    from . import neptune


if utils.modules_exist("comet-ml"):
    from . import comet

# General callbacks


class Save(_base.Operation):
    """Save best module according to specified metric.

    Example::

        import operator


        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                return loss


        step = TrainStep(criterion, device)
        iteration = tt.iterations.Train(step, module, dataloader)

        # Lower (operator.lt) loss than current best -> save the model
        iteration > tt.accumulators.Mean() > tt.callbacks.Save(
            module, "my_model.pt", comparator=operator.lt
        )

    Parameters
    ----------
    module: torch.nn.Module
        Module to save.
    path: pathlib.Path
        Path where module will be saved. Usually ends with `.pt` suffix,
        see PyTorch documentation.
    comparator: Callable(Number, Number) -> bool, optional
        Function comparing two values - current metric and best metric.
        If ``true``, save new module and use current value as the best one.
        One can use Python's standard operator library for this argument.
        Default: `operator.gt` (`current` > `best`)
    method: Callable(torch.nn.Module, pathlib.Path) -> None, optional
        Method to save `torch.nn.Module`. Takes module and path
        and returns anything (return value is discarded).
        Might be useful to transform model into
        `torch.jit.ScriptModule` or do some preprocessing before saving.
        Default: `torch.save` (whole model saving)
    log: str | int, optional
        Severity level for logging object's actions.
        Available levels of logging:
            NONE        0
            TRACE 	5
            DEBUG 	10
            INFO 	20
            SUCCESS 	25
            WARNING 	30
            ERROR 	40
            CRITICAL 	50
        Default: `NONE` (no logging, `0` priority)

    Arguments
    ---------
    data: Any
        Anything which can be passed to `comparator` (e.g. `torch.Tensor`).

    Returns
    -------
    Any
        Data passed in forward

    """

    def __init__(
        self,
        module: torch.nn.Module,
        path: pathlib.Path,
        comparator: typing.Callable = operator.gt,
        method: typing.Callable = None,
        log: typing.Union[str, int] = "NONE",
    ):
        self.module = module
        self.path = path
        self.comparator = comparator
        self.method = (
            lambda module, path: torch.save(module, path) if method is None else method
        )
        self.log = log

        self.best = None

    def forward(self, data: typing.Any) -> typing.Any:
        if self.best is None or self.comparator(data, self.best):
            self.best = data
            self.method(self.module, self.path)
            loguru.logger.log(
                self.log, "New best value: {}".format(self.best),
            )

        return data


class TimeStopping(_base.Operation):
    """Stop `epoch` after specified duration.

    Python's `time.time()` functionality is used.

    Can be placed anywhere (e.g. `step > TimeStopping(60 * 60)`) as it's
    not data dependent.

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                return loss


        step = TrainStep(criterion, device)
        iteration = tt.iterations.Train(step, module, dataloader)

        # Stop after 30 minutes
        iteration > tt.callbacks.TimeStopping(duration=60 * 30)

    Parameters
    ----------
    duration: int | float
        How long to run (in seconds) before exiting program.
    log: str | int, optional
        Severity level for logging object's actions.
        Available levels of logging:
            NONE        0
            TRACE 	5
            DEBUG 	10
            INFO 	20
            SUCCESS 	25
            WARNING 	30
            ERROR 	40
            CRITICAL 	50
        Default: `NONE` (no logging, `0` priority)

    Arguments
    ---------
    data: Any

    Returns
    -------
    Any
        Data passed in forward

    """

    def __init__(
        self, duration: float, log="NONE",
    ):
        self.duration = duration
        self.log = log
        self._start = time.time()

    def forward(self, data):
        if time.time() - self._start > self.duration:
            loguru.logger.log(
                self.log, "Stopping after {} seconds.".format(self.duration)
            )
            raise exceptions.TimeStopping()
        return data


class TerminateOnNan(_base.Operation):
    """Stop `epoch` if any `NaN` value encountered in `data`.

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                return loss, targets


        step = TrainStep(criterion, device)
        step > tt.Select(loss=0) > tt.callbacks.TerminateOnNan()

    Parameters
    ----------
    log: str | int, optional
        Severity level for logging object's actions.
        Available levels of logging:
            NONE        0
            TRACE 	5
            DEBUG 	10
            INFO 	20
            SUCCESS 	25
            WARNING 	30
            ERROR 	40
            CRITICAL 	50
        Default: `NONE` (no logging, `0` priority)

    Arguments
    ---------
    data: torch.Tensor
        Tensor possibly containing `NaN` values.

    Returns
    -------
    Any
        Data passed in forward

    """

    def __init__(
        self, log: typing.Union[str, int] = "NONE",
    ):
        self.log = log

    def forward(self, data):
        if torch.any(torch.isnan(data)):
            loguru.logger.log(self.log, "NaN values found, exiting with 1.")
            raise exceptions.TerminateOnNan()
        return data


class EarlyStopping(_base.Operation):
    """Stop `epoch` if `patience` was reached without improvement.

    Used to stop training if neural network's desired value didn't improve
    after `patience` steps.

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                return loss, accuracy


        step = TrainStep(criterion, device)
        iteration = tt.iterations.Train(step, module, dataloader)

        # Stop if accuracy does not improve for `5` iterations
        iteration > tt.Select(accuracy=1) > tt.accumulators.Mean() > tt.callbacks.EarlyStopping(
            patience=5
        )

    Parameters
    ----------
    patience: int
        How long not to terminate if metric does not improve
    delta: Number, optional
        Difference between `best` value and current considered as an improvement.
        Default: `0`.
    comparator: Callable(Number, Number) -> bool, optional
        Function comparing two values - current metric and best metric.
        If ``true``, reset patience and use current value as the best one.
        One can use Python's standard `operator` library for this argument.
        Default: `operator.gt` (`current` > `best`)
    log: str | int, optional
        Severity level for logging object's actions.
        Available levels of logging:
            NONE        0
            TRACE 	5
            DEBUG 	10
            INFO 	20
            SUCCESS 	25
            WARNING 	30
            ERROR 	40
            CRITICAL 	50
        Default: `NONE` (no logging, `0` priority)

    Arguments
    ---------
    data: Any
        Anything which can be passed to `comparator` (e.g. `torch.Tensor`).

    Returns
    -------
    Any
        Data passed in forward

    """

    def __init__(
        self,
        patience: int,
        delta: numbers.Number = 0,
        comparator: typing.Callable = operator.gt,
        log="NONE",
    ):
        self.patience = patience
        self.delta = delta
        self.comparator = comparator
        self.log = log
        self.best = None

        self._counter = -1

    def forward(self, data):
        if self.best is None or self.comparator(data, self.best):
            self._counter = -1
        else:
            self._counter += 1
        if self._counter == self.patience:
            loguru.logger.log(
                self.log, "Stopping early, best found: {}".format(self.best)
            )
            raise exceptions.EarlyStopping()


class Unfreeze(_base.Operation):
    """Unfreeze module's parameters after `n` steps.

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                return loss, accuracy


        step = TrainStep(criterion, device)
        iteration = tt.iterations.Train(step, module, dataloader)

        # Assume `module`'s parameters are frozen

        # Doesn't matter what data goes it, so you can unfreeze however you wish
        # And it doesn't matter what the accumulated value is
        iteration > tt.Select(accuracy=1) > tt.accumulators.Sum() > tt.callbacks.Unfreeze(
            module
        )

    Parameters
    ----------
    module: torch.nn.Module
        Module whose `parameters` will be unfrozen (`grad` set to `True`).
    n: int
        Module will be unfrozen after this many steps.
    log: str | int, optional
        Severity level for logging object's actions.
        Available levels of logging:
            NONE        0
            TRACE 	5
            DEBUG 	10
            INFO 	20
            SUCCESS 	25
            WARNING 	30
            ERROR 	40
            CRITICAL 	50
        Default: `NONE` (no logging, `0` priority)

    Arguments
    ---------
    data: Any

    Returns
    -------
    Any
        Data passed in forward

    """

    def __init__(self, module, n: int = 0, log="NONE"):
        self.module = module
        self.n = n
        self.log = log

        self._counter = -1

    def forward(self, data):
        self._counter += 1
        if self._counter == self.n:
            loguru.logger.log(self.log, "Unfreezing module's parameters")
            for param in self.module.parameters():
                param.requires_grad_(True)
        return data


class Log(_base.Operation):
    r"""Log data using `loguru.logger`.

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                return loss, accuracy


        step = TrainStep(criterion, device)
        iteration = tt.iterations.Train(step, module, dataloader)

        # Log with loguru.logger accuracy
        iteration > tt.Select(accuracy=1) > tt.callbacks.Logger("Accuracy")

    Parameters
    ----------
    name : str
        Name under which data will be logged.
        It will be in format "{name}: {data}"
    log : str | int, optional
        Severity level for logging object's actions.
        Available levels of logging:
            NONE        0
            TRACE 	5
            DEBUG 	10
            INFO 	20
            SUCCESS 	25
            WARNING 	30
            ERROR 	40
            CRITICAL 	50
        Default: `INFO`

    Arguments
    ---------
    data: Any
        Anything which can be sensibly represented with `__str__` magic method.

    Returns
    -------
    Any
        Data passed in forward

    """

    def __init__(self, name: str, log="INFO"):
        self.name = name
        self.log = log

    def forward(self, data):
        loguru.logger.log(self.log, "{}: {}".format(self.name, data))
        return data
