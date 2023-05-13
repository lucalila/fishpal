# Master Thesis on Comparing modular approaches for parameter-efficient fine-tuning

## Target of the Project:
Compare different parameter-efficient fine-tuning methods in the multi-task setting (GLUE data): LoRA layers, prefix-tuning, adapters. Calculate the baselines for these three approaches separately; use the FISH mask to determine which method / which parameters should be updated in which setting to reach the best results, and fine-tune a combined model by updating only those parameters that were selected by the FISH mask. Evaluate which method (baselines vs. combined) is performing best on the given datasets / tasks


### Basic source code retrieved from: 
https://github.com/varunnair18/FISH/tree/main
https://github.com/jxhe/unify-parameter-efficient-tuning/tree/25b44ac0e6f70e116af15cb866faa9ddc13b6c77

For ablation study 1, the following repo is needed as well:
https://github.com/morningmoni/UniPELT/tree/master

Another implementation of fish mask for glue that might be a good inspiration: https://github.com/RunxinXu/ChildTuning/blob/main/ChildTuningD.py
