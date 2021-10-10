## Language Modelling
___
This code base is adapted from this [repository](https://github.com/pytorch/fairseq) which the users should install. We further recommend the users to refer to [this README](https://github.com/pytorch/fairseq/tree/main/examples/language_model) for downloading and preprocessing the Wiki-103 data. We further use the pytorch implementation of [Universal Transformer](https://github.com/sarthmit/Universal_Transformers) implemented using [fairseq](https://github.com/pytorch/fairseq/). 
___
### Additions
We added the compositional attention mechanism [here](fairseq/modules/compositional_attention.py) in `fairseq/modules/compositional_attention.py` and further change certain details of the basic transformer layers, models and parser arguments to incorporate this mechanism. For details, refer to
- `fairseq/modules/compositional_attention.py`
- `fairseq/modules/compositional_attention_func.py`
- `fairseq/modules/transformer_layer.py`
- `fairseq/models/universal_transformer.py`
- `fairseq/models/universal_transformer_lm.py`
- `fairseq/models/transformer.py`
- `fairseq/models/transformer_lm.py`
___
### Training
We provide the `run_wiki.sh` script to perform training on the Wiki-103 dataset. The user must download and preprocess the data and have it in the `data-bin/wikitext-103` directory.

To run the models, we use

```./run_wiki.sh [type] [mode] [layers] [emb-dim] [ffn-dim] [attn-dim] [s-dim] [searches] [retrievals] [lr] [seed] [extras]```

- **type**: `Universal` for model with parameter sharing and `Stacked` for without.
- **model**: `Standard` for multi-head attention and `Compositional` for the proposed mechanism.
- **layers**: Number of layers/iterations for the model to run through.
- **emb-dim**: Encoding dimension for each token.
- **ffn-dim**: Feed-forward dimension for the Residual connection.
- **attn-dim**: Dimensions for retrievals, typically kept the same as **emb-dim**.
- **s-dim**: Dimensions for retrieval selection mechanism.
- **searches**: Number of searches for the model; heads for multi-head attention.
- **retrievals**: Number of retrievals for the proposed model.
- **lr**: Learning rate used to train the model.
- **seed**: Seed for reproducibility.
- **extras**: Further flags to give to the model. Possibilities include
  - `--qk-rule` for the dot-product mechanism for retrieval selection.
  - `--nonlinear` for using nonlinear Compositional Attention - MLP variant.

We test with `Universal` type of transformer and always use `--qk-rule` with our `Compositional` variant and train with `embed-dim` and `attn-dim` as 512, `ffn-dim` as 2048, `searches` and `retrievals` as 8, `s-dim` as 32, and train over 3 seeds.

For learning rate, we train one seed for each of the learning rates in `[0.0005, 0.001, 0.002, 0.004, 0.007]` and choose the model that has the best validation perplexity. This turns out to be `0.002` for the baseline multi-head attention and `0.004` for the proposed compositional attention. We then train the 3 seeds on these hyperparameters and plot the validation perplexity.
___
### Evaluation
For evaluating the learned language model, we provide the `eval_wiki.sh` script that needs to be provided with the `--path` for the model.