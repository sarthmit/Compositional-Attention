## Equilateral Triangle Detection
___
### Motivation
This is a task based on detecting the presence of equilateral triangle in the image. Each image has three clusters of points and the aim is to figure out if those clusters constitute an equilateral triangle; they do when the distance between the mid-points of any two pair of clusters is identical.
___
### Experiments
We provide the `run.sh` script for easy execution of the code. In particular, one can run the code with the following command.

```./run.sh [model] [iterations] [dim] [searches] [qk-dim] [retrievals] [lr] [seed] [extras]```

- **model**: Can be `Transformer` for standard multi-head attention and `Compositional` for compositional attention.
- **iterations**: The number of iterations/layers for the universal transformer to run.
- **dim**: The encoder dimension of the image patches.
- **searches**: The number of searches (heads for multi-head attention) in the model.
- **qk-dim**: The dimensionality for the retrieval queries and keys.
- **retrievals**: The number of retrievals in the model.
- **lr**: The learning rate for training.
- **seed**: Seed for reproducibility.
- **extras**: Further flags to give to the model. The possibilities include
  - `--dot` for dot-product attention based retrieval instantiation.

We typically run the proposed model with `--dot` provided. When not provided, then this leads to training of Compositional Attention - MLP ablation variant.

We encourage the reader to check out the paper for further details about the hyperparameters and to feel free to reach out in case of doubt.