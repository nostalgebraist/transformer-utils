import torch.nn
import transformers

_TORCH_NN_ORIGINAL = torch.nn
_TORCH_NN_ORIGINAL_LINEAR = _TORCH_NN_ORIGINAL.Linear

_TFM_PRETRAINED_MODEL_ORIGINAL = transformers.PreTrainedModel
_TFM_PRETRAINED_MODEL_INIT_WEIGHTS_ORIGINAL = transformers.PreTrainedModel.init_weights


def init_weights_without_init(self):
    # Prune heads if needed
    if self.config.pruned_heads:
        self.prune_heads(self.config.pruned_heads)

    # Tie weights if needed
    self.tie_weights()


class LazyLinearAPICompatible(torch.nn.LazyLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(out_features=out_features, bias=bias)


class LowMemoryLoadContext:
    def __enter__(self):
        torch.nn.Linear = LazyLinearAPICompatible
        transformers.PreTrainedModel.init_weights = init_weights_without_init

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.nn.Linear = _TORCH_NN_ORIGINAL_LINEAR
        transformers.PreTrainedModel.init_weights = _TFM_PRETRAINED_MODEL_INIT_WEIGHTS_ORIGINAL
        return exc_type is not None
