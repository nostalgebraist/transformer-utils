import torch
import transformers

from util.python_utils import make_print_if_verbose
from util.module_utils import get_child_module_by_names
from .load_context import LowMemoryLoadContext


def low_memory_load(config_path,
                    model_path,
                    config_cls=None,
                    model_cls=None,
                    high_memory_device="cuda:0",
                    verbose=True
                    ):
    vprint = make_print_if_verbose(verbose)

    if isinstance(high_memory_device, str):
        high_memory_device = torch.device(high_memory_device)

    if config_cls is None:
        config_cls = transformers.AutoConfig

    if model_cls is None:
        model_cls = transformers.AutoModelForCausalLM

    vprint("start")

    with LowMemoryLoadContext():
        state_dict = torch.load(
            model_path,
            map_location=high_memory_device,
        )

        vprint("loaded state dict")

        config = config_cls.from_pretrained(config_path)

        vprint("made config obj")

        # uses lazy init, no memory
        model = model_cls.from_config(config)

        vprint("made model obj")

        # START gpu --> cpu --> gpu handoff, one leaf module at a time
        handled = set()

        for name in dict(model.named_parameters()).keys():
            prefix = name.rpartition(".")[0]
            mod = get_child_module_by_names(model, prefix.split("."))

            if prefix in handled:
                continue

            vprint((name, prefix, mod))

            mk, uk, er = [], [], []
            mod._load_from_state_dict(
                state_dict,
                prefix=prefix + ".",
                local_metadata={},
                strict=True,
                missing_keys=mk,
                unexpected_keys=uk,
                error_msgs=er,
            )
            vprint((mk, uk, er))
            mod.to(high_memory_device)
            sdks = [k for k in state_dict if k.startswith(prefix)]
            for k in sdks:
                del state_dict[k]
            handled.add(prefix)

        # END gpu --> cpu --> gpu handoff, one leaf module at a time

        vprint("loaded params into memory")

        # does the buffers
        model = model.to(high_memory_device)

        vprint("loaded all into memory")

        model.eval()

    return model
