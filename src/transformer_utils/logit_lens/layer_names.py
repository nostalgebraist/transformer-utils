from ..util.module_utils import get_child_module_by_names


def make_layer_names(
    model,
    block_step=1,
    include_input=True,
    force_include_output=True,
    include_subblocks=False,
    decoder_layer_names: list = ['final_layernorm', 'lm_head']
):
    h = get_child_module_by_names(model.base_model, ["h"])
    h_names = [f"h.{i}" for i in range(len(h))]

    last_h_name = h_names[-1]

    h_names = h_names[::block_step]
    if force_include_output and last_h_name not in h_names:
        h_names.append(last_h_name)

    if include_subblocks:
        names = [sub_name for name in h_names for sub_name in (f"{name}.attn", name)]
    else:
        names = h_names

    if include_input:
        names = ["input"] + names

    def _subset(a, b):
        return a == b or a.startswith(b + ".")

    def _names_overlap(a, b):
        return _subset(a, b) or _subset(b, a)

    names = [name for name in names
             if not any([_names_overlap(name, dname) for dname in decoder_layer_names])
             ]

    return names
