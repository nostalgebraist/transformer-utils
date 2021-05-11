import torch.nn
import transformers
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter

_TORCH_NN_ORIGINAL_LINEAR = torch.nn.Linear

_TFM_PRETRAINED_MODEL_INIT_WEIGHTS_ORIGINAL = (
    transformers.modeling_utils.PreTrainedModel.init_weights
)
_TFM_CONV1D_ORIGINAL = transformers.modeling_utils.Conv1D


def init_weights_without_init(self):
    pass


class LazyLinearAPICompatible(torch.nn.LazyLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(out_features=out_features, bias=bias)


class LazyTransformersConv1D(LazyModuleMixin, _TFM_CONV1D_ORIGINAL):
    cls_to_become = _TFM_CONV1D_ORIGINAL
    weight: UninitializedParameter

    def __init__(self, nf, nx):
        super().__init__(nf=nf, nx=0)
        self.nx = 0
        self.weight = UninitializedParameter()

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.nx != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.nx = input.shape[-1]
                self.weight.materialize((self.nf, self.nx))
                self.reset_parameters()


class LowMemoryLoadContext:
    def __enter__(self):
        torch.nn.Linear = LazyLinearAPICompatible
        transformers.modeling_utils.Conv1D = LazyTransformersConv1D
        transformers.PreTrainedModel.init_weights = init_weights_without_init

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.nn.Linear = _TORCH_NN_ORIGINAL_LINEAR
        transformers.modeling_utils.Conv1D = _TFM_CONV1D_ORIGINAL
        transformers.PreTrainedModel.init_weights = (
            _TFM_PRETRAINED_MODEL_INIT_WEIGHTS_ORIGINAL
        )
        return exc_type is None
