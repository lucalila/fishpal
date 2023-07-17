#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

# modified run_glue.py file for combination of petl, unipelt and sparse updates from fish paper

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import json  # Todo: new import from other script, may not be relevant

import numpy as np
from datasets import load_dataset, load_metric
import decimal  # added this for max operation
import math  # added for log calc

import transformers
from transformers import (
    AdapterConfig,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    # MultiLingAdapterArguments,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import torch
from torch.utils.data import DataLoader

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
from utils import freeze_params, choose_gpu, freeze_params_by_layers

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


# TODO#### BEGIN OF INSERTION ###########

# Todo: If small nb of labels, the first one can be used,
#  otherwise use the second one. For sst-2 and mnli the first one can be used.
def calculate_the_importance_label(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute)
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        # create a gradient dictionary with 0s only for all model parameter tensors
        # adjusted: only calc gradients for module layers and classifier layer
        if "classifier" in name or "adapter" in name or "lora" in name or "prefix" in name:
            gradients_dict[name] = torch.zeros_like(param).to(cuda_device)
    # choose the calculation method
    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    # select a subset of the training data, according to the hyperparameter
    for idx, inputs in enumerate(data_loader):
        if idx >= num_samples:
            break

        # print(idx)

        inputs.pop("idx", None)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(cuda_device)

        # From the paper: A given entry in F_weights relates to the average of the square gradient of the model’s
        # output with respect to a given parameter.

        # get the model output (logits + loss)
        return_dicts = model(**inputs)
        # extract the loss
        loss = return_dicts["loss"]

        # calculate the gradients
        loss.backward()

        for name, param in model.named_parameters():
            # grad method is either square or absolute
            # add the squared value of each model param to the gradients dict (which is initialized by 0)
            # todo: adapted for all three modules
            if "classifier" in name or "adapter" in name or "lora" in name or "prefix" in name:
                # calculate the square of the gradients
                gradients_dict[name] += grad_method(param.grad).data
        # set the gradients to zero
        model.zero_grad()
    # return the dict of the squares of the gradients
    return gradients_dict


def calculate_the_importance_expect(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute)
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    for idx, inputs in enumerate(data_loader):
        if idx >= num_samples:
            break

        # print(idx)

        inputs.pop("idx", None)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(cuda_device)

        return_dicts = model(**inputs)

        logits = return_dicts["logits"]

        log_probs = torch.nn.functional.log_softmax(logits, -1)
        probs = torch.nn.functional.softmax(logits, -1)

        for b in range(logits.shape[0]):
            for i in range(logits.shape[1]):
                loss = - log_probs[b, i]
                loss.backward(retain_graph=True)

                prob = probs[b, i]

                for name, param in model.named_parameters():
                    gradients_dict[name] += (prob * grad_method(param.grad)).data

                model.zero_grad()

    return gradients_dict


def create_mask_gradient(model, train_dataset, data_collator, num_samples, keep_ratio, sample_type, grad_type):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=data_collator,
        shuffle=True
    )

    if sample_type == "label":
        importance_method = calculate_the_importance_label  # for small number of labels this one
    elif sample_type == "expect":
        importance_method = calculate_the_importance_expect  # for large number of labels this one needs to be used
    else:
        raise NotImplementedError

    # this returns the fish mask
    gradients = importance_method(model, data_loader, num_samples, cuda_device, grad_type)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0
    all_params_size = 0

    classifier_mask_dict = {}

    # iterate over fish mask
    for k, v in gradients.items():  # todo: do not use all gradient items -> only for the modules
        # don't count classifier layer, they should be all trainable
        if "classifier" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))  # tensors now includes the fish mask values

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size

    assert keep_num > 0  # make sure there are trained params

    top_pos = torch.topk(tensors, keep_num)[1]  # get the positions with the highest fisher information

    masks = torch.zeros_like(tensors, device=cuda_device)  # create a mask of zeros

    masks[top_pos] = 1  # only add ones for the tops positions -> only these weights will be updated

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
        now_idx = end_idx

    assert now_idx == len(masks)

    # Add the classifier's mask to mask_dict
    mask_dict.update(classifier_mask_dict)

    model.to(original_device)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    pretrain_weight_size = 0

    for k, v in mask_dict.items():
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            pretrain_weight_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    print(pretrain_weight_size, classifier_size, all_params_size)
    print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

    return mask_dict


def evaluate_modules_per_layer(gradients, layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                               prefix_scaling=10000,
                               adapter_scaling=5):
    total_grad_sum_prefix, total_grad_sum_adapter, total_grad_sum_lora = 0, 0, 0  # todo: just for checking
    total_nb_params_prefix, total_nb_params_adapter, total_nb_params_lora = 0, 0, 0  # todo: just for checking

    sum_prefix, num_prefix = 0, 0
    collect_results = dict()

    # preformat prefix weights
    for k, v in gradients.items():
        if "prefix" in k:
            if "roberta.encoder.prefix_embed_MLP.2.weight" not in k and \
                    "roberta.encoder.prefix_embed_MLP.2.bias" not in k:
                # print("These are the params in the first if: ", k)
                total_grad_sum_prefix += v.sum()  # todo: just for checking
                total_nb_params_prefix += torch.numel(v)  # todo: just for checking
                #print(k, v.size())  # todo: just for checking
                sum_prefix += v.sum()
                num_prefix += torch.numel(v)
            elif "roberta.encoder.prefix_embed_MLP.2.weight" == k:
                total_grad_sum_prefix += v.sum()  # todo: just for checking
                total_nb_params_prefix += torch.numel(v)  # todo: just for checking
                #print(k, v.size())  # todo: just for checking
                v = v.view(-1, 12 * 2, 12, 64)  # split the weights according to the model implementation
                print("This is the size of v: ", v.size())
                v = v.permute([1, 2, 0, 3])
                print("This is the permuted size of v: ", v.size())
                mlp_weights = v.split(2)
                print("This is the length of MLP weights: ", len(mlp_weights))
                for i in mlp_weights:
                    print("These are the new weight elements shape: ", i.size())
            elif "roberta.encoder.prefix_embed_MLP.2.bias" == k:
                total_grad_sum_prefix += v.sum()  # todo: just for checking
                total_nb_params_prefix += torch.numel(v)  # todo: just for checking
                #print(k, v.size())  # todo: just for checking
                v = v.view(12 * 2, 12, 64)
                mlp_bias = v.split(2)

    # todo: this is just a check
    # sum up per layer:
    #for weight, bias in zip(mlp_weights, mlp_bias):
    #    print("Sum of the layer: ", weight.sum()+bias.sum()+sum_prefix)
    
    for layer in layers:
        layer_sum, layer_num = 0, 0  # init by 0
        layer_sum += mlp_weights[layer].sum()  # per layer add the respective weight sum
        layer_sum += mlp_bias[layer].sum()  # per layer add the respective bias sum todo: should I remove this?
        layer_sum += sum_prefix  # add the sum of the other (not-layer-specific) params
        # nb of params
        layer_num += torch.numel(mlp_weights[layer])
        layer_num += torch.numel(mlp_bias[layer])
        layer_num += num_prefix

        if layer in collect_results:
            collect_results[layer]["prefix"] = decimal.Decimal((layer_sum / layer_num).item() * prefix_scaling)
        else:
            collect_results[layer] = {"prefix": decimal.Decimal((layer_sum / layer_num).item() * prefix_scaling)}

    group_param_per_layer = dict()
    for layer in layers:
        params_per_layer = []
        named_layer = "layer." + str(layer) + "."
        for key in gradients.keys():
            if named_layer in key:
                params_per_layer.append(key)
        group_param_per_layer[layer] = params_per_layer

    # sum of params and nb of params for normalization
    for layer, params in group_param_per_layer.items():
        sum_adapter, sum_lora, num_adapter, num_lora = 0, 0, 0, 0
        for param in params:
            if "adapter" in param:  # and "bias" not in param:
                total_grad_sum_adapter += gradients[param].sum()  # todo: just for checking
                total_nb_params_adapter += torch.numel(gradients[param])  # todo: just for checking
                #print(param, gradients[param].size())
                #print("Adapter: ", layer, param)
                #print("Sum: ", gradients[param].sum())
                #print("Nb: ", torch.numel(gradients[param]))
                num_adapter += torch.numel(gradients[param])
                sum_adapter += gradients[param].sum()
            elif "lora" in param and "lora_A" not in param:
                total_grad_sum_lora += gradients[param].sum()  # todo: just for checking
                total_nb_params_lora += torch.numel(gradients[param])  # todo: just for checking
                #print(param, gradients[param].size())
                num_lora += torch.numel(gradients[param])
                sum_lora += gradients[param].sum()
        collect_results[layer]["adapter"] = decimal.Decimal((sum_adapter / num_adapter).item() * adapter_scaling)
        collect_results[layer]["lora"] = decimal.Decimal((sum_lora / num_lora).item())

    best_module_per_layer = dict()

    print("------Evaluation")
    print("---Prefix")
    print("Total Sum: ", total_grad_sum_prefix)
    print("Total Nb Params: ", total_nb_params_prefix)
    print("Normalized Sum: ", total_grad_sum_prefix/total_nb_params_prefix)

    print("---Adapter")
    print("Total Sum: ", total_grad_sum_adapter)
    print("Total Nb Params: ", total_nb_params_adapter)
    print("Normalized Sum: ", total_grad_sum_adapter / total_nb_params_adapter)

    print("---Lora")
    print("Total Sum: ", total_grad_sum_lora)
    print("Total Nb Params: ", total_nb_params_lora)
    print("Normalized Sum: ", total_grad_sum_lora / total_nb_params_lora)

    for key, mod_values in collect_results.items():
        item = [sec_key for sec_key, sec_value in mod_values.items() if sec_value == max(mod_values.values())]
        best_module_per_layer[key] = item[0]

    return best_module_per_layer


def prepare_params_to_exclude(modules_per_layer, prefix_tuning_active, task_name, classifier_training_active=True):
    lora_items = [".attention.self.query.lora_B", ".attention.self.value.lora_B",
                  ".attention.self.query.lora_A", ".attention.self.value.lora_A"]
    if task_name == "sst2":
        adapter_items = [".output.adapters.sst2.adapter_down.0.weight",
                         ".output.adapters.sst2.adapter_down.0.bias",
                         ".output.adapters.sst2.adapter_up.weight",
                         ".output.adapters.sst2.adapter_up.bias"]
    elif task_name == "mrpc":
        adapter_items = [".output.adapters.mrpc.adapter_down.0.weight",
                         ".output.adapters.mrpc.adapter_down.0.bias",
                         ".output.adapters.mrpc.adapter_up.weight",
                         ".output.adapters.mrpc.adapter_up.bias"]
    else:  # mnli
        adapter_items = [".output.adapters.mnli.adapter_down.0.weight",
                         ".output.adapters.mnli.adapter_down.0.bias",
                         ".output.adapters.mnli.adapter_up.weight",
                         ".output.adapters.mnli.adapter_up.bias"]
    prefix_items = ["roberta.encoder.prefix_enc_embed.weight",
                     "roberta.encoder.prefix_embed_MLP.0.bias",
                     "roberta.encoder.prefix_embed_MLP.0.weight",
                     "roberta.encoder.prefix_embed_MLP.2.weight",
                     "roberta.encoder.prefix_embed_MLP.2.bias"]

    except_para_l = []
    for k, v in modules_per_layer.items():
        layer_id = "layer." + str(k)
        if "lora" in v:
            for par in lora_items:
                except_par = layer_id + par
                except_para_l.append(except_par)
        elif "adapter" in v:
            for par in adapter_items:
                except_par = layer_id + par
                except_para_l.append(except_par)
    if classifier_training_active: # per default classifier weights are updated
        except_para_l.append("classifier")
    if prefix_tuning_active: # if active, add prefix tuning params to exclude from freezing
        print("yes it is active")
        except_para_l.extend(prefix_items)

    return except_para_l


def create_mask_one_module(model, train_dataset, data_collator, num_samples, task_name,
                           grad_type="square",
                           classifier_training_active=True):
    print("Classifier training is active? ", classifier_training_active)
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=data_collator,
        shuffle=True
    )

    importance_method = calculate_the_importance_label  # for small number of labels this one

    # this returns the fish mask
    gradients = importance_method(model, data_loader, num_samples, cuda_device, grad_type)

    classifier_size = 0

    classifier_mask_dict = {}

    # FishSCALP implementation here
    # todo: comment in for evaluation, not needed atm, config is known
    #modules_per_layer = evaluate_modules_per_layer(gradients)  # evaluate which module is best for which layer
    #print(modules_per_layer)

    # todo: hard-coded final config to run experiments per task
    # mrpc combination
    modules_per_layer = {0: 'prefix',
                         1: 'adapter',
                         2: 'lora',
                         3: 'lora',
                         4: 'adapter',
                         5: 'adapter',
                         6: 'adapter',
                         7: 'adapter',
                         8: 'adapter',
                         9: 'adapter',
                         10: 'lora',
                         11: 'lora'}

    # sst2 combination
    modules_per_layer = {0: 'prefix',
                         1: 'prefix',
                         2: 'prefix',
                         3: 'prefix',
                         4: 'prefix',
                         5: 'adapter',
                         6: 'adapter',
                         7: 'adapter',
                         8: 'adapter',
                         9: 'adapter',
                         10: 'lora',
                         11: 'lora'}
            
    print(modules_per_layer)

    # get all layers in which prefix module is needed
    prefix_layers = [k for k, v in modules_per_layer.items() if v == 'prefix']
    adapter_layers = [k for k, v in modules_per_layer.items() if v == 'adapter']
    lora_layers = [k for k, v in modules_per_layer.items() if v == 'lora']
    print("These are the prefix layers: ", prefix_layers)
    print("These are the adapter layers: ", adapter_layers)
    print("These are the lora layers: ", lora_layers)

    # set prefix tuning to active if there is a prefix layer, so that gradients are calculated for the prefix params
    if len(prefix_layers) < 1:
        prefix_tuning_active = False
    else:
        prefix_tuning_active = True
    # prepare params that should not be frozen
    # todo: experiment with frozen classifier layer
    except_para_l = prepare_params_to_exclude(modules_per_layer,
                                              prefix_tuning_active,
                                              task_name,
                                              classifier_training_active)
    # print("Except params: ", except_para_l)
    freeze_params(model, except_para_l=except_para_l)  # function freezes all params, except the ones in the list
    for name, par in model.named_parameters():  # doublecheck
        print(name, par.requires_grad)

    mask_dict = {}

    for k, v in gradients.items():
        # comment from fish: don't count classifier layer, they should be all trainable
        if "classifier" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        elif "adapter" in k:
            for layer in adapter_layers:
                if "layer." + str(layer) + "." in k:  # set to 1 if it is an adapter layer
                    mask_dict[k] = torch.ones_like(v, device=cuda_device)  # mask for param is added
                    break
                else:  # set to 0 if it isn't an adapter layer
                    mask_dict[k] = torch.zeros_like(v, device=cuda_device)  # mask for param is added
        elif len(lora_layers) > 0 and "lora" in k:
            for layer in lora_layers:
                if "layer." + str(layer) + "." in k:
                    mask_dict[k] = torch.ones_like(v, device=cuda_device)  # mask for param is added
                    break
                else:
                    # print("Do I get into the lora else path?")
                    mask_dict[k] = torch.zeros_like(v, device=cuda_device)  # mask for param is added
        elif len(prefix_layers) > 0 and "prefix" in k:  # check if there is prefix tuning involved
            if "roberta.encoder.prefix_embed_MLP.2.weight" not in k and \
                    "roberta.encoder.prefix_embed_MLP.2.bias" not in k:
                # prefix params need to be trained, if they occur in any layer
                mask_dict[k] = torch.ones_like(v, device=cuda_device)  # mask for param is added
            elif "roberta.encoder.prefix_embed_MLP.2.weight" in k:
                # reshape tensor so that the correct layers are masked
                v = torch.zeros_like(v, device=cuda_device)
                v = v.view(-1, 12 * 2, 12, 64)  # split the weights according to the model implementation
                v = v.permute([1, 2, 0, 3])
                v = v.split(2)
                v = list(v)
                for layer in prefix_layers:
                    v[layer] = torch.ones_like(v[layer])  # add 1s for the prefix layers
                # revert reshaping
                v = torch.cat(tuple(v))
                v = v.permute([2, 0, 1, 3])
                mask_dict[k] = v.reshape(18432, 512)  # mask for param is added
                #print(mask_dict[k])
                #print("How many 1s in the mask dict - does it make sense for prefix tuning? ", mask_dict[k].sum())
                #print("How many params in total: ", torch.numel(mask_dict[k]))
            elif "roberta.encoder.prefix_embed_MLP.2.bias" in k:
                # reshape tensor so that the correct layers are masked
                #print("Size of the prefix tuning params before: ", v.size())
                v = torch.zeros_like(v, device=cuda_device)
                v = v.view(12 * 2, 12, 64)  # split the bias according to the model implementation
                v = list(v.split(2))
                for layer in prefix_layers:
                    v[layer] = torch.ones_like(v[layer])
                # revert reshaping
                v = torch.cat(tuple(v))
                #v = v.reshape(18432)
                mask_dict[k] = v.reshape(18432)  # mask for param is added
                #print("Size of the prefix tuning params after: ", mask_dict[k].size())
        else:
            #print("Else k's: ", k)
            mask_dict[k] = torch.zeros_like(v, device=cuda_device)  # add 0s for all other module params,
            # mask for param is added

    # Add the classifier's mask to mask_dict
    # todo: do not add the classifier mask, if classifier training is set to false
    if classifier_training_active:
        mask_dict.update(classifier_mask_dict)

    model.to(original_device)

    # Print the parameters for checking
    """
    classifier_size = 0
    all_params_size = 0
    pretrain_weight_size = 0

    for k, v in mask_dict.items():
        print("keys in gradients: ", k)
        print(v)
    
        #print("keys in mask: ", k)
        #print(v)
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            pretrain_weight_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    print(pretrain_weight_size, classifier_size, all_params_size)
    print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")
    """

    return mask_dict


def create_mask_random(model, train_dataset, data_collator, num_samples, keep_ratio):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    gradients = {}
    for name, param in model.named_parameters():
        gradients[name] = torch.rand(param.shape).to(original_device)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0
    all_params_size = 0

    classifier_mask_dict = {}

    for k, v in gradients.items():
        # don't count classifier layer, they should be all trainable
        if "classifier" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size

    assert keep_num > 0

    top_pos = torch.topk(tensors, keep_num)[1]

    masks = torch.zeros_like(tensors, device=original_device)

    masks[top_pos] = 1

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
        now_idx = end_idx

    assert now_idx == len(masks)

    # Add the classifier's mask to mask_dict
    mask_dict.update(classifier_mask_dict)

    model.to(original_device)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    pretrain_weight_size = 0

    for k, v in mask_dict.items():
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            pretrain_weight_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    print(pretrain_weight_size, classifier_size, all_params_size)
    print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

    return mask_dict


def create_mask_bias(model, train_dataset, data_collator, num_samples, keep_ratio):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    mask_dict = {}

    for name, param in model.named_parameters():
        if "classifier" in name:
            mask_dict[name] = torch.ones_like(param, device=original_device)
        elif "bias" in name:
            mask_dict[name] = torch.ones_like(param, device=original_device)
        else:
            mask_dict[name] = torch.zeros_like(param, device=original_device)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    bias_params_size = 0

    for k, v in mask_dict.items():
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            bias_params_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    print(bias_params_size, classifier_size, all_params_size)

    print(f"trainable parameters: {(bias_params_size + classifier_size) / all_params_size * 100} %")

    model.to(original_device)

    return mask_dict


class SparseUpdateTrainer(Trainer):
    def __init__(self, *args, mask, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)

        # mask out the gradients
        for name, params in self.model.named_parameters():

            device = params.device
            if name in self.mask:  # todo: I added this here -> not all params occur in my mask implementation
                # print(name, "  - mask")
                self.mask[name] = self.mask[name].to(device)
                params.grad.data.copy_(params.grad.data * self.mask[name].data)
                # print(params.grad.data)

        return loss

##### END OF FISH INSERTION

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    max_tokens_per_batch: Optional[int] = field(
        default=0,
        metadata={
            "help": "dynamic batching. Override batch size when larger than 0"
        },
    ),
    early_stopping_patience: Optional[int] = field(
        default=10,
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    load_adapter_config: bool = field(
        metadata={"help": "Whether the model is saved normally of with adapter config separately"}
    )

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_load_path_adapter: str = field(
        default="",
        metadata={"help": ""}
    )
    ### INSERTED BELOW
    # model_load_path_second: str = field(
    #    default="",
    #    metadata={"help": ""}
    # )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    # todo: INSERTED HERE (Unipelt)
    # prefix-tuning parameters
    add_enc_prefix: bool = field(
        default=False,
        metadata={"help": "Whether use prefix tuning"},
    )
    add_dec_prefix: bool = field(
        default=False,
        metadata={"help": "Whether use prefix tuning"},
    )
    add_cross_prefix: bool = field(
        default=False,
        metadata={"help": "Whether use prefix tuning"},
    )
    prefix_len: Optional[int] = field(
        default=10,
        metadata={"help": "length of prefix tokens"},
    )
    mid_dim: Optional[int] = field(
        default=512,
        metadata={"help": "dim of middle layer"},
    )
    # bitfit parameters
    tune_bias: bool = field(
        default=False,
        metadata={"help": "Whether tune bias terms"},
    )
    # LoRA parameters
    add_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use lora for linear layers"},
    )
    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "rank of lora"},
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "scaling = alpha / r"},
    )

    drop_first_layers: Optional[int] = field(
        default=0,
        metadata={
            "help": "drop first k layers, work for both prefix and adapter, freeze transformer layers if fine-tuning"},
    )
    drop_first_adapter_layers: Optional[int] = field(
        default=0,
        metadata={"help": "drop first k adapter layers"},
    )
    drop_first_prefix_layers_enc: Optional[int] = field(
        default=0,
        metadata={"help": "drop first k prefix layers"},
    )
    drop_first_prefix_layers_dec: Optional[int] = field(
        default=0,
        metadata={"help": "drop first k prefix layers"},
    )
    drop_first_prefix_layers_cross: Optional[int] = field(
        default=0,
        metadata={"help": "drop first k prefix layers"},
    )
    add_adapter_gate: bool = field(
        default=True,
        metadata={"help": "add a gate to the adapter"},
    )
    add_prefix_gate: bool = field(
        default=True,
        metadata={"help": "add a gate to the prefix"},
    )
    add_lora_gate: bool = field(
        default=True,
        metadata={"help": "add a gate to the lora"},
    )
    add_central_gate: bool = field(
        default=False,
        metadata={"help": "add a shared gate"},
    )
    # todo: END OF INSERTION unipelt


# todo INSERTED FROM petl.options
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    min_length: Optional[int] = field(
        default=10,
        metadata={
            "help": "minimal generation length"
        },
    )

    max_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "max generation length"
        },
    )

    num_beams: Optional[int] = field(
        default=5,
        metadata={
            "help": "minimal generation length"
        },
    )

    no_repeat_ngram_size: Optional[int] = field(
        default=0,
        metadata={
            "help": "minimal generation length"
        },
    )

    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "length penalty"
        },
    )


@dataclass
class TuneArguments:
    attn_mode: Optional[str] = field(
        default="none",
        metadata={
            "choices": ["prefix", "prefix_nomlp",
                        "none", "bitfit", "lora", "adapter",
                        "prompt_tuning"], \
 \
            "help": "config for attention, none to disable; \
                prefix: mlp reparameterization to output prefix P; \
                prefix_nomlp: prefix P as learned params; \
                adapter: adapter mode; \
                bitfit: the bitfit baseline; \
                lora: the lora baseline; \
                prompt_tuning: the prompt tuning baseline",
        },
    )

    attn_option: Optional[str] = field(
        default="concat",
        metadata={
            "choices": ["none",
                        "concat",
                        "cross_attn",
                        "cross_attn_noln",
                        "cross_attn_relu",
                        "parallel",
                        "sequential",
                        ], \
 \
            "help": "specific attn configs; \
                concat: concat prefix to self, this is prefix tuning baseline; \
                cross_attn_noln: prefix tuning with vanilla add composition (instead of gated add), \
                    need to be used together with 'attn_composition=add'; \
                cross_attn: cross_attn_noln plus a layernorm layer \
                cross_attn_relu: basically multi-head adapter, need to be used under 'prefix' mode; \
                parallel: parallel insertion form; need to be used under 'adapter' mode; \
                sequential: sequential insertion form; need to be used under 'adapter' mode;",

        },
    )

    attn_composition: Optional[str] = field(
        default="add",
        metadata={
            "choices": ["add", "gate_add"],
            "help": "the composition function \
                add: vanilla adding; \
                gate_add: gated adding like prefix tuning"
        },
    )

    ffn_mode: Optional[str] = field(
        default="none",
        metadata={
            "choices": ["adapter", "none", "lora"],

            "help": "config for ffn, none to disable; \
            adapter: adapter mode; \
            lora: the lora baseline",
        },
    )

    ffn_option: Optional[str] = field(
        default="none",
        metadata={
            "choices": ["parallel", "sequential", "pfeiffer", "none"], \
 \
            "help": "specific ffn configs; \
                parallel: parallel insertion form; \
                sequential: sequential insertion form; \
                pfeiffer: the Pfeiffer adapter config"
        },
    )

    ffn_adapter_layernorm_option: Optional[str] = field(
        default="in",
        metadata={
            "choices": ["in", "out", "none"],
            "help": "ffn adapter layernorm options; \
                none: no layernorm; \
                in: layernorm applied to input; \
                out: layernorm applied to output"
        },
    )

    ffn_adapter_init_option: Optional[str] = field(
        default="bert",
        metadata={
            "choices": ["bert", "lora"],
            "help": "ffn adapter option"
        },
    )

    ffn_adapter_scalar: Optional[str] = field(
        default="1",
        metadata={
            "help": "the scaling hyperparam for scaled adding composition; \
                set to 'learnable_scalar' to learn this as a parameter"
        },
    )

    mid_dim: Optional[int] = field(
        default=800,
        metadata={
            "help": ""
        },
    )

    attn_bn: Optional[int] = field(
        default=200,
        metadata={
            "help": "the attention bottleneck dimension"
        },
    )

    ffn_bn: Optional[int] = field(
        default=-1,
        metadata={
            "help": "the ffn bottleneck dimension"
        },
    )

    prefix_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": ""
        },
    )

    unfreeze_params: Optional[str] = field(
        default="ef_",
        metadata={
            "help": "param names that contain the string will \
                be unfreezed, all other params will be freezed"
        },
    )

    load_path: Optional[str] = field(
        default="",
        metadata={
            "help": ""
        },
    )

    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={
            "help": "scaling: alpha / r"
        },
    )

    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "scaling: alpha / r"
        },
    )

    lora_init: Optional[str] = field(
        default="lora",
        metadata={
            "choices": ["bert", "lora"],
            "help": ""
        },
    )


# todo: new class here
@dataclass
class SparseUpdateTrainingArguments:
    num_samples: int = field(
        default=1024,
        metadata={"help": "The number of samples to compute parameters importance"}
    )
    keep_ratio: float = field(
        default=0.005,
        metadata={"help": "The trainable parameters to total parameters."}
    )
    mask_method: str = field(
        default="label-absolute",
        metadata={"help": "The method to select trainable parameters. Format: sample_type-grad_type, \
                   where sample_type in \{label, expect\}, and grad_type in \{absolute, square\}"}
    )
    normal_training: bool = field(
        default=False,
        metadata={"help": "Whether to use typical BERT training method."}
    )
    mask_path: str = field(
        default="",
        metadata={"help": "The path for existing mask."}
    )
    data_split_path: str = field(
        default="",
        metadata={"help": "The path for existing training data indices."}
    )


@dataclass
class MBARTArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    dropout: Optional[float] = field(
        default=0.3,
        metadata={
            "help": ""
        },
    )

    attention_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": ""
        },
    )


from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdapterArguments:
    """
    The subset of arguments related to model training.
    """

    train_adapter: bool = field(default=False, metadata={"help": "Train an model instead of the full model."})
    load_adapter: Optional[str] = field(
        default="", metadata={"help": "Pre-trained model module to be loaded from Hub."}
    )
    adapter_config: Optional[str] = field(
        default="pfeiffer", metadata={"help": "Adapter configuration. Either an identifier or a path to a file."}
    )
    adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the model configuration."}
    )
    adapter_reduction_factor: Optional[int] = field(
        default=None, metadata={"help": "Override the reduction factor of the model configuration."}
    )
    language: Optional[str] = field(default=None, metadata={"help": "The training language, e.g. 'en' for English."})


@dataclass
class MultiLingAdapterArguments(AdapterArguments):
    """
    Arguemnts related to model training, extended by arguments for multilingual setups.
    """

    load_lang_adapter: Optional[str] = field(
        default=None, metadata={"help": "Pre-trained language model module to be loaded from Hub."}
    )
    lang_adapter_config: Optional[str] = field(
        default=None, metadata={"help": "Language model configuration. Either an identifier or a path to a file."}
    )
    lang_adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the language model configuration."}
    )
    lang_adapter_reduction_factor: Optional[int] = field(
        default=None, metadata={"help": "Override the reduction factor of the language model configuration."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               SparseUpdateTrainingArguments,
                               MultiLingAdapterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, sparse_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, sparse_args, adapter_args = parser.parse_args_into_dataclasses()
        # tune_args,

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                        test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # todo: INSERTION from unipelt
    # additional arguments
    config.add_enc_prefix = model_args.add_enc_prefix
    config.add_dec_prefix = model_args.add_dec_prefix
    config.add_cross_prefix = model_args.add_cross_prefix
    config.prefix_len = model_args.prefix_len
    config.mid_dim = model_args.mid_dim
    if 'bert' in model_args.model_name_or_path:
        num_layers = config.num_hidden_layers
    elif 'bart' in model_args.model_name_or_path:
        num_layers = config.encoder_layers
    config.add_adapter_gate = model_args.add_adapter_gate
    config.add_prefix_gate = model_args.add_prefix_gate
    config.tune_bias = model_args.tune_bias
    config.add_lora = model_args.add_lora
    config.lora_r = model_args.lora_r
    config.lora_alpha = model_args.lora_alpha
    config.add_lora_gate = model_args.add_lora_gate
    config.add_central_gate = model_args.add_central_gate
    config.early_stopping_patience = data_args.early_stopping_patience

    if model_args.drop_first_layers == 0:
        config.drop_first_prefix_layers_enc = list(range(model_args.drop_first_prefix_layers_enc))
        config.drop_first_prefix_layers_dec = list(range(model_args.drop_first_prefix_layers_dec))
        config.drop_first_prefix_layers_cross = list(range(model_args.drop_first_prefix_layers_cross))
    else:
        # override by drop_first_layers
        model_args.drop_first_adapter_layers = model_args.drop_first_layers
        config.drop_first_prefix_layers_enc = list(range(model_args.drop_first_layers))
        config.drop_first_prefix_layers_dec = list(range(model_args.drop_first_layers - num_layers))
        config.drop_first_prefix_layers_cross = list(range(model_args.drop_first_layers - num_layers))
    # todo: INSERTION END unipelt

    setattr(training_args, 'max_tokens_per_batch', data_args.max_tokens_per_batch)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     num_labels=num_labels,
    #     finetuning_task=data_args.task_name,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )

    #if model_args.load_adapter_config:
    #     # todo: first load pre-trained roberta,
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    # # todo: then add pre-trained adapter to it
    # model_adapter.load_adapter(model_args.model_load_path_adapter)
    #else:
        # # todo: regular loading
        # print("Do I get here?")
        # config = AutoConfig.from_pretrained(
        #     model_args.config_name if model_args.config_name else model_args.model_load_path_adapter,
        #     num_labels=num_labels,
        #     finetuning_task=data_args.task_name,
        #     cache_dir=model_args.cache_dir,
        #     revision=model_args.model_revision,
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )
        #
        # model = AutoModelForSequenceClassification.from_pretrained(
        #     model_args.model_name_or_path,
        #     from_tf=bool(".ckpt" in model_args.model_load_path_adapter),
        #     config=config_adapter,
        #     cache_dir=model_args.cache_dir,
        #     revision=model_args.model_revision,
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # todo: INSERTION FROM unipelt
    # Setup adapters
    if adapter_args.train_adapter:
        task_name = data_args.task_name or "glue"
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters:
            print("Is the task name not in the model config adapters?")
            # resolve the adapter config
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
                leave_out=list(range(model_args.drop_first_adapter_layers))
            )
            # load a pre-trained from Hub if specified
            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter,
                    config=adapter_config,
                    load_as=task_name,
                )
            # otherwise, add a fresh adapter
            else:
                model.add_adapter(task_name, config=adapter_config)
        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = AdapterConfig.load(
                adapter_args.lang_adapter_config,
                non_linearity=adapter_args.lang_adapter_non_linearity,
                reduction_factor=adapter_args.lang_adapter_reduction_factor,
            )
            # load the language adapter from Hub
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                config=lang_adapter_config,
                load_as=adapter_args.language,
            )
        else:
            lang_adapter_name = None
        # Freeze all model weights except of those of this adapter
        model.train_adapter([task_name])
        # Set the adapters to be used in every forward pass
        if lang_adapter_name:
            model.set_active_adapters([lang_adapter_name, task_name])
        else:
            model.set_active_adapters([task_name])
    else:
        except_para_l = []
        if config.tune_bias:
            except_para_l.append('bias')
        if config.add_lora:
            except_para_l.append('lora')
        if any([config.add_enc_prefix, config.add_dec_prefix, config.add_cross_prefix]):
            except_para_l.append('prefix')
        if len(except_para_l) > 0:
            freeze_params(model, except_para_l=except_para_l)
        elif model_args.drop_first_layers > 0:
            freeze_params_by_layers(model, num_layers, num_frozen_layers=model_args.drop_first_layers)

        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter training"
            )

        # todo: END OF INSERTION HERE

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable_params: {trainable_params}, total_params: {total_params}")

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # print(model)
    # for n, p in model.named_parameters():
    #    print(n, p.requires_grad)
    # print("This is the config: ", config)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    if sparse_args.normal_training:

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            do_save_full_model=True,  # otherwise, P in AP may not be saved
            do_save_adapters=adapter_args.train_adapter
        )
    else:
        # todo: insertion from Fish mask
        if sparse_args.mask_path != "":
            mask = torch.load(sparse_args.mask_path, map_location="cpu")
        else:
            if sparse_args.mask_method == "bias":
                mask_method = create_mask_bias
                mask = create_mask_bias(
                    model, train_dataset, data_collator, sparse_args.num_samples, sparse_args.keep_ratio
                )

            elif sparse_args.mask_method == "random":
                mask_method = create_mask_random

                mask = create_mask_random(
                    model, train_dataset, data_collator, sparse_args.num_samples, sparse_args.keep_ratio
                )
            # todo: my implementation
            elif sparse_args.mask_method == "one_module":
                mask_method = create_mask_one_module

                mask = create_mask_one_module(
                    model, train_dataset, data_collator, sparse_args.num_samples, data_args.task_name)
                # this is to check how many 1s are in the mask (actual trained params)
                actual_trained_params = 0
                for k, v in mask.items():
                    actual_trained_params += v.sum()
                    #print(k)
                    #print(v)
                print("Nb of trained params: ", actual_trained_params)

            else:
                sample_type, grad_type = sparse_args.mask_method.split("-")

                mask = create_mask_gradient(
                    model,
                    train_dataset,
                    data_collator,
                    sparse_args.num_samples,
                    sparse_args.keep_ratio,
                    sample_type,
                    grad_type
                )

            #print("This is the mask: \n", mask)
            # todo: This below was commented out also before
                # def reset_classifier(model):
                #     for n, p in model.named_parameters():
                #         if "classifier.weight" in n:
                #             p.data.normal_(mean=0.0, std=model.config.initializer_range)
                #         elif "classifier.bias" in n:
                #             p.data.zero_()

                # reset_classifier(model)
        
        # Initialize our Trainer
        #todo: set up trainer as before
        trainer = SparseUpdateTrainer(
            model=model,
            args=training_args,
            mask=mask,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            do_save_full_model=True,  # otherwise, P in AP may not be saved
        )

    for k, v in model.named_parameters():
        print(k)
        print(v.size())
    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    #print("After training", model)

    #todo: inserted
    if not sparse_args.normal_training:
       torch.save(mask, os.path.join(training_args.output_dir, "mask.bin"))  # saves the mask

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        #print(model)
        for n, _ in model.named_parameters():
            print(n)

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        print(eval_datasets)
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset.remove_columns_("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
