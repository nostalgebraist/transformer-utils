import torch
import transformers

from ..util.python_utils import make_print_if_verbose
from ..util.module_utils import get_child_module_by_names
from ..util.tfm_utils import normalize_inconsistent_state_dict_keys
from .load_context import LowMemoryLoadContext


DEFAULT_GENERIC_INPUT = torch.as_tensor([[0]])


def modify_weights_after_load(model):
    """the part of PreTrainedModel.init_weights that isn't initializing weights"""
    # Prune heads if needed
    if model.config.pruned_heads:
        model.prune_heads(model.config.pruned_heads)

    # Tie weights if needed
    model.tie_weights()


def low_memory_load(
    config_path,
    model_path,
    config_cls=None,
    model_cls=None,
    high_memory_device="cuda:0",
    generic_input=DEFAULT_GENERIC_INPUT,
    verbose=True,
):
    vprint = make_print_if_verbose(verbose)

    if isinstance(high_memory_device, str):
        high_memory_device = torch.device(high_memory_device)

    if config_cls is None:
        config_cls = transformers.AutoConfig

    vprint("start")

    with LowMemoryLoadContext():
        config = config_cls.from_pretrained(config_path)

        vprint("made config obj")

        state_dict = torch.load(
            model_path,
            map_location=high_memory_device,
        )

        state_dict = normalize_inconsistent_state_dict_keys(state_dict)

        vprint("loaded state dict")

        # uses lazy init, no memory
        if model_cls is None:
            model = transformers.AutoModelForCausalLM.from_config(config)
        else:
            model = model_cls(config=config)

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

        # does stuff like weight tying, now that the weights actually exist
        modify_weights_after_load(model)

        model.eval()

        # ensures we materialize the lazy params (and delete the hooks for doing so), before doing anything else
        #
        # if you add pre-hooks before doing this step, you get OrderedDict mutation exceptions
        with torch.no_grad():
            out = model(generic_input.to(high_memory_device))
            out = None

    return model
