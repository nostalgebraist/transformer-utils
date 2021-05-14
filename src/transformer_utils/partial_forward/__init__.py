import traceback

from ..util.python_utils import make_print_if_verbose


class AfterStoppingPointException(Exception):
    pass


def add_partial_forward_hooks(model, verbose=False, debug=False, output_names=None):
    vprint = make_print_if_verbose(verbose)
    dprint = make_print_if_verbose(debug)

    can_skip = output_names is not None

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

    def _record_to_sink_hook(module, input, output) -> None:
        if hasattr(model, "_output_sink_names"):
            this_name = module._partial_forward_name
            dprint(f"reached output of {repr(this_name)}")
            dprint(f"model._output_sink_names: {model._output_sink_names}")

            if this_name in model._output_sink_names:
                dprint(f"{repr(this_name)} in sink")

                model._output_sink[this_name] = output

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
    add_partial_forward_hooks(model, verbose=verbose, debug=debug, output_names=output_names)

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
        traceback.clear_frames(e.__traceback__)

    del model._output_sink_names

    return_val = model._output_sink
    del model._output_sink

    return return_val
