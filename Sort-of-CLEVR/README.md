## Relational Reasoning in Sort-of-CLEVR
___
### Motivation
This is a Visual Question Answering (VQA) task that aims to test the model's capacity to perform relational reasoning. In particular, this dataset consists of the following three types of questions.
- **Unary**: Questions based on the properties of only single objects.
- **Binary**: Questions based on the relationships between two objects.
- **Ternary**: Questions based on the relationships between three objects.
This is a suite of four tasks that are based on logical reasoning of different flavours. Each sub-task in this suite has a rule or pattern associated with it and the models are tasked with uncovering that and applying the same rule even on new previously unseen objects, thus allowing for OoD testing of models.
___
### Experiments
For baselines, we compare with standard multi-head attention and showcase the empirical differences.

The data for Sort-of-CLEVR can be generated through the `sort_of_clevr_generator.py` but we also provide the version of the data that we used [here](https://drive.google.com/drive/folders/1WyGVDGEJq7ImLt7nIWIbVjM92L0Wxe1C?usp=sharing)

We provide the `run.sh` script for easy execution of the code. In particular, one can run the code with the following command.

```./run.sh [model] [iterations] [dim] [searches] [qk-dim] [retrievals] [seed] [extras]```

- **model**: Can be `Transformer` for standard multi-head attention and `Compositional` for compositional attention.
- **iterations**: The number of iterations/layers for the universal transformer to run.
- **dim**: The encoder dimension of the image patches.
- **searches**: The number of searches (heads for multi-head attention) in the model.
- **qk-dim**: The dimensionality for the retrieval queries and keys.
- **retrievals**: The number of retrievals in the model.
- **seed**: Seed for reproducibility.
- **extras**: Further flags to give to the model. The possibilities include
  - `--dot` for dot-product attention based retrieval instantiation.

We typically run the proposed model with `--dot` provided. When not provided, then this leads to training of Compositional Attention - MLP ablation variant.

We encourage the reader to check out the paper for further details about the hyperparameters and to feel free to reach out in case of doubt.