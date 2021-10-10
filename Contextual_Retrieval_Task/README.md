## Contextual Retrieval Task
___
### Motivation
This task is based on retrieving relevant information based on certain dynamic and contextual retrieval cues. This directly tests a model's ability to flexibly retrieve different information depending on the context that is provided.
___
### Experiments
For baselines, we compare with standard multi-head attention and showcase the empirical differences as well as qualitative analyses.

We provide the `run.sh` script for easy execution of the code. In particular, one can run the code with the following command.

```./run.sh [model] [seq-len] [dim] [search-dim] [retrieval-dim] [searches] [gt-searches] [retrievals] [gt-retrievals] [seed] [extras]```

- **model**: Can be `Standard` for Multi-head attention and `Compositional-dot` for Compositional attention.
- **seq-len**: Cardinality of the input set.
- **dim**: Encoder dimension for each element of the set.
- **search-dim**: Total dimensionality associated with the number of searches in the model.
- **retrieval-dim**: Total dimensionality associated with the number of retrievals in the model.
- **searches**: Number of searches in the model.
- **gt-searches**: Number of ground-truth searches in the data.
- **retrievals**: Number of retrievals in the model.
- **gt-retrievals**: Number of ground truth retrievals in the data.
- **seed**: Seed for reproducibility.
- **extras**: Further flags to give to the model. Possibilities include
  - `--concat` to include current token's information in the MLP after attention.
  - `--no-coeff` for the data output to be based on non-coefficient based sum.
  - `--ood` for training and testing on the OoD setup.

We typically keep the `seq-len` as 10 and always provide `--concat` to the model to give it the retrieval switching context. We test the model with and without `--ood` and ablate over various searches and retrievals as well as the different dimensions. We typically keep the different dimensions as the same, and test across the values `32, 64, 128, 256 and 512` for different search-retrieval task variants.

We encourage the reader to check out the paper for further details about the hyperparameters and to feel free to reach out in case of doubt.