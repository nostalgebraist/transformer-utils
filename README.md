## transformer-utils

Utilities for the HuggingFace [transformer](https://github.com/huggingface/transformers/) library, focused on loading and using large pretrained autoregressive language models like GPT-2 and GPT-Neo.

This package is unofficial and not associated with HuggingFace.

- Lets you load large (~2.7B) models in low-resource environments like Google Colab
- Interpreting models with the "logit lens"
  - For background, see 
    - ["interpreting GPT: the logit lens"](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) by nostalgebraist
    - ["Finding the Words to Say: Hidden State Visualizations for Language Models"](https://jalammar.github.io/hidden-states/) by Jay Alammar

### Example usage

#### Loading in low-memory environments

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
