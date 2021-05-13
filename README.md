## transformer-utils

Utilities for the HuggingFace [transformer](https://github.com/huggingface/transformers/) library, focused on loading and using large pretrained autoregressive language models like GPT-2 and GPT-Neo.

This package is unofficial and not associated with HuggingFace.

- Lets you load large (~2.7B) models in low-resource environments like Google Colab
- Interpreting models with the "logit lens"
  - For background, see
    - ["interpreting GPT: the logit lens"](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) by nostalgebraist
    - ["Finding the Words to Say: Hidden State Visualizations for Language Models"](https://jalammar.github.io/hidden-states/) by Jay Alammar

### Example usage

#### Load in a low-memory environment

Loading a 2.7B model:

```python
from transformer_utils.low_memory import enable_low_memory_load

enable_low_memory_load()

model = transformers.AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
```

This works fine in an ordinary (non-Pro) Google Colab notebook, with ~12 GB RAM and a T5 GPU.

Inference will work up to the full context window length of 2048 tokens without memory issues.

#### Logit lens

```python
import torch
import transformers
from transformer_utils.low_memory import enable_low_memory_load

enable_low_memory_load()

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
model = transformers.AutoModelForCausalLM.from_pretrained('gpt2-xl')

def text_to_input_ids(text):
    toks = tokenizer.encode(text)
    return torch.as_tensor(toks).view(1, -1).cuda()

input_ids = text_to_input_ids("This is an example. You can probably think of a more fun text to use than this one.")

plot_logit_lens(model, tokenizer, input_ids, start_ix=0, end_ix=45)  # logits

plot_logit_lens(model, tokenizer, input_ids, start_ix=0, end_ix=45, probs=True)  # probabilities
```

#### Get activations from any part of the model

###### ...and without running parts you don't need

```python
from transformer_utils.partial_forward import partial_forward

output = partial_forward(
    model=model,  # your model
    output_names=[
        'h.0',  # output of the 1st layer
        'h.2.attn.c_attn',  # query/key/value matrix from the 3rd laer
        'h.5.mlp.c_proj',   #  feed-forward activations from the 6th layer
    ],
    input_ids  # the input to run
)

# each of these is a tensor
output['h.0']
output['h.2.attn.c_attn']
output['h.5.mlp.c_proj']
```

For efficiency, `partial_forward` doesn't run any part of the model later than the ones you specify in `output_names`.

For example, suppose `model` above was GPT-2 XL.  Then it has 48 layers.  But the forward pass in the code above stops running after the 6th layer of 48 -- so the compute and memory cost is far lower than a full `model.forward`.

This makes it easy to write new "heads" that do further computation on the model's activations.  Examples:

###### Using the first two layers of a model as input  features to a new one

```python
from transformer_utils.partial_forward import partial_forward


output = partial_forward(
    model, ['h.0', 'h.1',], input_ids  # the input to run
)

# each of these is a tensor
output['h.0']
output['h.2.attn.c_attn']
output['h.5.mlp.c_proj']
```
