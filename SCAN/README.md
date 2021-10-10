## SCAN Task
___
This code base is adapted from EMNLP 2021 paper [The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers](https://arxiv.org/abs/2108.12284).
Please follow [this README](https://github.com/RobertCsordas/transformer_generalization) instructions for setting up the experiments. We use the SCAN task to test our proposed model.

### Additions
___
We just added our model, *Compositional Attention*, to this code base. The description of the attention mechanism can be found [here](layers/transformer/compositional_attention.py) in `layers/transformer/compositional_attention.py` and the implementation of the full model can be found [here](layers/transformer/compositional_transformer.py) in `layers/transformer/compositional_transformer.py`.

### Experiments
___
To run SCAN Dataset experiments of cut off length 30, run ``` wandb sweep --entity username --project project_name --name scan_trafo_30  sweeps_compositional/scan_trafo_30.yaml```. This will create a sweep and returns a `sweep_id`. 
Then run ```wandb agent username/project_name/sweep_id```. This will start experiments of all the models mentioned in the `.yaml` file for 5 seeds. 


### Evaluations
___
Edit config file ```paper/config.json```. Enter your project name in the field "wandb_project" (e.g. "username/project_name").

Run ```python -W ignore plot_big_result_table.py``` to get the results in the latex format. 