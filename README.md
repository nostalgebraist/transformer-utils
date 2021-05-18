## transformer-utils

Utilities for the HuggingFace [transformers](https://github.com/huggingface/transformers/) library, focused on loading and using large pretrained autoregressive language models like GPT-2 and GPT-Neo.

This package is unofficial and not associated with HuggingFace.

Features:

- Load large (~2.7B) models in low-resource environments like Google Colab
- Get activations from any part of the model, without running parts you don't need
- Interpret models with the "logit lens"
  - For background, see
    - ["interpreting GPT: the logit lens"](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) by nostalgebraist
    - ["Finding the Words to Say: Hidden State Visualizations for Language Models"](https://jalammar.github.io/hidden-states/) by Jay Alammar

## Example usage

### Load in a low-memory environment

Loading a 2.7B model:

```python
from transformer_utils.low_memory import enable_low_memory_load

enable_low_memory_load()

model = transformers.AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
```

This works fine in an ordinary (non-Pro) Google Colab notebook, with ~12 GB RAM and a T5 GPU.

Inference will work up to the full context window length of 2048 tokens without memory issues.

### Logit lens

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

plot_logit_lens(model, tokenizer, input_ids, start_ix=0, end_ix=45, kl=True)  # K-L divergence
```

You can do also some other things that aren't in the original blog posts.  This will break down the transformer blocks into their attention and MLP parts:

```python
plot_logit_lens(model, tokenizer, input_ids, start_ix=0, end_ix=45, include_subblocks=True)
```

You can also change the definition of the "decoder" to include some of the later blocks/subblocks of the model.  This helps especially in interpreting GPT-Neo hidden states.

```python
# assume we have a 48-layer model
# so 'h.47' is the final layer

# include last layer in decoder
plot_logit_lens(
    model, tokenizer, input_ids, start_ix=0, end_ix=45,
    decoder_layer_names=['h.47', 'final_layernorm', 'lm_head']
)

# include just the last MLP subblock in decoder
plot_logit_lens(
    model, tokenizer, input_ids, start_ix=0, end_ix=45,
    decoder_layer_names=['h.47.mlp', 'final_layernorm', 'lm_head']
)
```

### Get activations from any part of the model

###### ...and without running parts you don't need

```python
from transformer_utils.partial_forward import partial_forward

output = partial_forward(
    model=model,  # your `transformers` model
    output_names=[
        'h.0',  # output of the 1st layer
        'h.2.attn.c_attn',  # query/key/value matrix from the 3rd layer
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

This makes it easy to write new "heads" that do further computation on the model's activations.

Some examples:

##### Using the first two layers of a model as features extractors for binary classification

```python
output_names=['h.0', 'h.1',]
classifier_hidden_size=768

feature_vector_size = base_model.config.n_embd * len(output_names)

classifier = nn.Sequential(
    nn.Linear(feature_vector_size, classifier_hidden_size),
    nn.ReLU(),
    nn.Linear(classifier_hidden_size, 2),
)

opt = torch.optim.Adam(classifier.parameters())

for input_ids, targets in dataset:  # `dataset` is your classification train data
    with torch.no_grad():
        hidden_states = partial_forward(
            base_model,
            output_names,
            input_ids,
        )

    # shape (batch, sequence, len(output_names) * model's hidden size)
    feature_vector = torch.cat(
        [hidden_states[name] for name in output_names],
        dim=-1
    )

    # shape (batch, sequence, 2)
    classifier_out = classifier(feature_vector)

    # simple avg pool over sequence dim -- in practice find attention works well for this step :)
    # shape (batch, 2)
    logits = classifier_out.mean(dim=1)

    loss = F.cross_entropy(target=targets, input=logits)
    loss.backward()
    opt.step()
    opt.zero_grad()
```


##### Finetuning the first two layers of a model

This is exactly the same as the above, with just two changes:

- Remove the `with torch.no_grad()` wrapper around `partial_forward`
- Optimize the base model's params too:
  - `opt = torch.optim.Adam(list(classifier.parameters()) + list(base_model.parameters()))`

If you want to train a model like these ones for real use, I recommend writing a custom `nn.Module`.  [See here](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/fd96e9482186f5dbeaa27bd6179087c892c577d6/selector_model/selector_nn_neo.py#L263) for an example.
