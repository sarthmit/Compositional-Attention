## Logical Reasoning in ESBN Tasks
___
### Motivation
This is a suite of four tasks that are based on logical reasoning of different flavours. Each sub-task in this suite has a rule or pattern associated with it and the models are tasked with uncovering that and applying the same rule even on new previously unseen objects, thus allowing for OoD testing of models.
___
### Experiments
For baselines, we compare with standard multi-head attention and showcase the empirical differences.

For ease of running the experiments, we provide the users with `scripts/run.sh` that contains the different model setups that we run on this suite. In particular, we compare the standard multi-head attention with our proposed compositional attention ablated over multiple possible retrievals.

We encourage the reader to check out the paper for further details about the hyperparameters and to feel free to reach out in case of doubt.