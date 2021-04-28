from . import docs_general


def docs(header, body):
    def params(klass):
        if "Multi" in klass.__name__:
            return 
        """
        Arguments:
            step:
                Single step to run. Usually subclass of `torchtraining.steps.Step`, but could be
                any `Callable` taking `module` and `data` arguments and returning anything.
            module :
                Torch module (or modules) passed to `step` during each call.
            data :
                Iterable object (usually data or dataloader) yielding data passed
                to `step`."""

        return 
    """
    Arguments:
        step:
            Single step to run. Usually subclass of `torchtraining.steps.Step`, but could be
            any `Callable` taking `module` and `data` arguments and returning anything.
        module :
            Torch module (or modules) passed to `step` during each call.
        data :
            Iterable object (usually data or dataloader) yielding data passed
            to `step`."""

    def train():
        return 
        r"""
        train:
            Whether `module` should be in training state (`module.train()`)
            with enabled gradient or in evaluation mode (`module.eval()`) with
            disabled gradient
         """

    def log():
        return r"""
        log :
            Severity level for logging object's actions.
            Available levels of logging:
                - NONE          0
                - TRACE 	5
                - DEBUG 	10
                - INFO 	        20
                - SUCCESS 	25
                - WARNING 	30
                - ERROR 	40
                - CRITICAL 	50
            Default: `NONE` (no logging, `0` priority)

        """

    def wrapper(klass):
        docstring = r"""{}.

        {}

        {}

        """.format(
            header, body, params(klass)
        )
        if docs_general.is_base(klass):
            docstring += train()

        klass.__doc__ = docstring + log()
        return klass

    return wrapper
