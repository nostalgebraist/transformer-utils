import warnings
import inspect

from ..util.python_utils import make_print_if_verbose


PARTIAL_FORWARD_FORCE_FALSE_KWARGS = {
    "use_cache",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
}

PARTIAL_FORWARD_FORCE_FALSE_KWARGS_MSG = """`partial_forward` was passed the argument {kwarg} but will ignore it.

`partial_forward` ignores arguments that configure output shape in `transformers`, since its output shape is configured entirely through the `output_names` argument."""

VALIDATE_OUTPUT_BASE_MODEL_MSG = """Some `output_names` were not found on the model (a `{model_class_name}`), but exist on its base model (a `{base_model_class_name}`).

Try either passing `model.base_model` as the model, OR adding the string '{base_model_prefix}.' to the start of each output name.

Names not found: {names}"""

VALIDATE_OUTPUT_NOT_FOUND_MSG = """Some `output_names` were not found on the model.

To see valid output names, try `dict(model.named_modules()).keys()`.

Names not found: {names}"""


class AfterStoppingPointException(Exception):
    pass


def _validate_output_names(model, output_names):
    if output_names is None:
        return

    findable_names = dict(model.named_modules()).keys()

    findable_names_base_model = set()
    if hasattr(model, "base_model") and hasattr(model, "base_model_prefix"):
        findable_names_base_model = dict(model.base_model.named_modules()).keys()

    problem_names = [name for name in output_names if name not in findable_names]

    base_model_names = [
        name for name in problem_names if name in findable_names_base_model
    ]

    if len(base_model_names) > 0:
        raise ValueError(
            VALIDATE_OUTPUT_BASE_MODEL_MSG.format(
                model_class_name=model.__class__.__name__,
                base_model_class_name=model.base_model.__class__.__name__,
                base_model_prefix=model.base_model_prefix,
                names=base_model_names,
            )
        )

    if len(problem_names) > 0:
        raise ValueError(VALIDATE_OUTPUT_NOT_FOUND_MSG.format(names=problem_names))


def add_partial_forward_hooks(model, verbose=False, debug=False, output_names=None):
    vprint = make_print_if_verbose(verbose)
    dprint = make_print_if_verbose(debug)

    _validate_output_names(model, output_names)

    can_skip = output_names is not None
    can_skip = can_skip and hasattr(model, "_partial_forward_force_false_kwargs")

    names_to_mods = {}
    indices_to_names = {}
    names, mods = [], []
    for i, (name, mod) in enumerate(model.named_modules()):
        if hasattr(mod, "_partial_forward_name") and mod._partial_forward_name != name:
            can_skip = False

        mod._partial_forward_name = name
        indices_to_names[i] = name

        names.append(name)
        mods.append(mod)
        names_to_mods[name] = mod

        if output_names is not None:
            should_have_hook = name in output_names
            already_has_hook = hasattr(mod, "_record_to_sink_handle")
            can_skip = can_skip and (should_have_hook == already_has_hook)

    if can_skip:
        dprint("already have partial forward hooks, skipping")
        return

    sig = inspect.signature(model.__class__.forward)
    model._partial_forward_force_false_kwargs = (
        PARTIAL_FORWARD_FORCE_FALSE_KWARGS.intersection(sig.parameters.keys())
    )

    def _record_to_sink_hook(module, input, output) -> None:
        if hasattr(model, "_output_sink_names"):
            this_name = module._partial_forward_name
            dprint(f"reached output of {repr(this_name)}")
            dprint(f"model._output_sink_names: {model._output_sink_names}")

            if this_name in model._output_sink_names:
                dprint(f"{repr(this_name)} in sink")

                to_record = output
                if isinstance(to_record, tuple) and len(to_record) == 1:
                    to_record = to_record[0]

                model._output_sink[this_name] = to_record

            if all([name in model._output_sink for name in model._output_sink_names]):
                dprint("have all model._output_sink_names, stopping")

                raise AfterStoppingPointException

    for name, mod in zip(names, mods):
        if hasattr(mod, "_record_to_sink_handle"):
            vprint(f"clearing existing handle at {repr(name)}")
            mod._record_to_sink_handle.remove()

        if output_names is None or name in output_names:
            rts_handle = mod.register_forward_hook(_record_to_sink_hook)
            mod._record_to_sink_handle = rts_handle


def partial_forward(
    model,
    output_names,
    *args,
    verbose=False,
    debug=False,
    **kwargs,
):
    vprint = make_print_if_verbose(verbose)

    add_partial_forward_hooks(
        model, verbose=verbose, debug=debug, output_names=output_names
    )

    for k in model._partial_forward_force_false_kwargs:
        if kwargs.get(k):
            warnings.warn(PARTIAL_FORWARD_FORCE_FALSE_KWARGS_MSG.format(kwarg=repr(k)))
        kwargs[k] = False

    model._output_sink_names = output_names

    if hasattr(model, "_output_sink"):
        vprint("clearing existing _output_sink")
        for v in model._output_sink.values():
            del v
        del model._output_sink

    model._output_sink = {}

    try:
        model(*args, **kwargs)
    except AfterStoppingPointException as e:
        pass

    del model._output_sink_names

    return_val = model._output_sink
    del model._output_sink

    return return_val
