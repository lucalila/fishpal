# Master Thesis on ''Comparing Modular Approaches for Parameter-Efficient Fine-Tuning''

## Abstract:
Large language models (LLMs) exhibit impressive natural language understanding across tasks. 
As language model size increases, adapting them to specific tasks becomes computationally expensive. 
In-context learning has been proposed as an alternative to standard fine-tuning of LLMs. 
However, prompting generally underperforms standard fine-tuning. 
Also, finding the best prompts is not straightforward, as the process is brittle e.g., to the wording of the prompt and the number of examples. 
To tackle these issues, parameter- efficient fine-tuning (PEFT) has been proposed. 
This paradigm adds modular components to a pre-trained model; these are fine-tuned on the target task while the LLM is not updated. 
PEFT components have dedicated per-task capacity and permit updating a model without forgetting previous knowledge, while a composition of such modules can improve the multi-task capabilities of an LLM. 
PEFT can reach the performance of standard fine-tuning. This has motivated research in this area and a plethora of methods have recently been proposed. 
However, to evaluate which modular approach is suitable for a set of tasks, experimentation with selected modular approaches per task is needed. 
This often requires an exhaustive search over methods and hyperparameters, which is difficult in practice. 
This study proposes a new criterion, based on the Fisher information matrix, to select which PEFT approach to use to adapt an LLM to a specific task. 
The novel a priori Fisher-informed selection of Prefix-tuning, Adapters, and LoRA for Transformers, FishPAL, avoids costly training experiments and only trains one combination per task. 
In the experiments of this thesis, FishPAL consistently outperforms the baselines on different GLUE tasks while updating only 2-4% of the total model parameters and adding only 0.4% of the base model’s parameters during inference.
## Source Code
* https://huggingface.co/docs/transformers/index
* https://github.com/adapter-hub/adapter-transformers
* https://github.com/varunnair18/FISH/tree/main
* https://github.com/morningmoni/UniPELT/tree/master
## Reproduction of Results:
### Baselines
##### Adapter
* exps/run_glue_adapter.sh (SST-2)
* exps/run_glue_adapter_mnli.sh
* exps/run_glue_adapter_mrpc.sh
##### LoRA
* exps/run_glue_lora.sh (SST-2)
* exps/run_glue_lora_mnli.sh 
* exps/run_glue_lora_mrpc.sh
##### Prefix-Tuning
* exps/run_glue_prefix.sh (SST-2)
* exps/run_glue_prefix_mnli.sh
* exps/run_glue_prefix_mrpc.sh
### FishPAL
* exps/run_glue_sparse_sst2.sh (Seeds: 111, 11, 22)
* exps/run_glue_sparse_mnli.sh (Seeds: 11, 22, 88)
* exps/run_glue_sparse_mrpc.sh (Seeds: 11, 22, 88)

## Notes:
All experiments were run on 1 Google Cloud GPU (NVIDIA T4 GPU with 4vCPUs and 15GB
RAM in the zone us-west3-b)


