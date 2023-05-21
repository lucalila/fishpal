#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=glue
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --time=0
#SBATCH --array=0-4%5


#datasets=$1 # we use $TASK_NAME instead
#seed=$2 # we use seed in this script
# TODO: Insertion from fish
keep_ratio=0.005  # Todo: Integrate below, new param
mask_num_samples=1024  # Todo: integrate below, new param
method="label-square" # Todo: integrate below, new param
normal_training=True  # todo: normally set to false to trigger sparse updates
# TODO: END insertion from fish


export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model

cache_dir=${TRANSFORMERS_CACHE}


TASK_NAME=mnli
metric="accuracy"

# wandb env variables
export WANDB_PROJECT=glue.${TASK_NAME}
export WANDB_WATCH="false"

DATE=`date +%Y%m%d`


seed=42


# set to 1 for debug mode which only
# uses 1600 training examples
debug=0  # OK  #todo: set to 1 for trials

# set to "wandb" to use weights & bias
report_to="none" # this is just a visualization tool

bsz=16  # 2 for lmu gpus, orig 16
gradient_steps=1


model="roberta-base" # todo: roberta-base
lr=1e-4 # todo: Adapter: 1e-4; Lora: 5e-4; Prefix: 2e-4
num_train_epochs=10 # todo: 50 for unipelt (7 in fish paper; 10 from He et al.)
max_seq_length=128

max_grad_norm=1  # OK (this is the default from huggingface)
weight_decay=0.1  # OK (this is the RoBERTa default)
warmup_updates=0  # OK (this is the default)
warmup_ratio=0.06  # OK (this is from He et al., default is 0.0)
max_steps=-1  # OK (this is the default from huggingface and overwritten from epochs and batch size)
max_tokens_per_batch=0  #
#max_train_samples=100 # ADJUSTED -> 500, 1000, all # K = {100, 500, 1000 , all} from paper, all has the best performance (from unipelt)

lr_scheduler_type="polynomial"  # OK (this is from He et al.)
unfreeze='ef_'
max_eval_samples=1600 # OK
logging_steps=50 # OK

eval_strategy="epoch" # OK - same for fish mask
save_steps=5000 # OK

### UNIPELT params ##########
# set to True for Prefix
add_enc_prefix=False
add_dec_prefix=False
add_cross_prefix=False
prefix_len=10
mid_dim=512
# set to True for BitFit
tune_bias=False
# set to True for LoRA
add_lora=False
lora_r=8
lora_alpha=16
drop_first_layers=0
drop_first_adapter_layers=0
drop_first_prefix_layers_enc=0
drop_first_prefix_layers_dec=0
drop_first_prefix_layers_cross=0
add_adapter_gate=False  # ONLY ADAPTER BASELINE HERE
add_prefix_gate=False  # ONLY ADAPTER BASELINE HERE
add_lora_gate=False  # ONLY ADAPTER BASELINE HERE
add_central_gate=False
# set to True for adapters -> what about the bottleneck dim? -> check model.config.adapters
train_adapter=True  # ONLY ADAPTER BASELINE HERE
#load_adapter
adapter_config=pfeiffer  # default is pfeiffer
adapter_non_linearity=None
adapter_reduction_factor=None
#language
load_lang_adapter=None
lang_adapter_config=None
lang_adapter_non_linearity=None
lang_adapter_reduction_factor=None
early_stopping_patience=10
##### UNIPELT END #######


extra-cmd=""
#extra_cmd="--max_train_samples ${max_train_samples}"  # ADJUSTED
debug_str=""

# this is only for debugging
if [ "${debug}" = 1 ];
then
    weight_decay=0.1
    max_grad_norm=1
    max_train_samples=10
    max_eval_samples=10
    bsz=2 # batch-size
    gradient_steps=1
    num_train_epochs=1
    max_steps=-1
    eval_strategy='steps'
    save_steps=10
    report_to="none"
    logging_steps=10
    extra_cmd="--max_train_samples ${max_train_samples} --max_predict_samples 150"
    debug_str=".debug"
fi

exp_name=glue.${TASK_NAME}.model_${model}.pre_${add_enc_prefix}.lora_${add_lora}.adap_${train_adapter}
exp_name+=.preg_${add_prefix_gate}.lorag_${add_lora_gate}.adapg_${add_adapter_gate}
exp_name+=.adapc_${adapter_config}.bsz_${bsz}.epoch_${num_train_epochs}.lr_${lr}

#exp_name+=.fl_${ffn_adapter_layernorm_option}.finit_${ffn_adapter_init_option}
#exp_name+=.fs_${ffn_adapter_scalar}.unfrz_${unfreeze}.ne${num_train_epochs}
exp_name+=.seed_${seed}.${debug_str}
SAVE=checkpoints/glue/${TASK_NAME}/${DATE}/${exp_name}
echo "${SAVE}"
rm -rf ${SAVE}; mkdir -p ${SAVE}

rm checkpoints/hf_model/downloads/*.lock
rm checkpoints/hf_model/*.lock


# python -m torch.distributed.launch --nproc_per_node 2 --master_port=${port} examples/pytorch/text-classification/run_glue.py \

# roberta-base
python -u examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path ${model} \
    --task_name $TASK_NAME \
    --do_train True \
    --do_eval False\
    --do_predict False \
    --num_samples ${mask_num_samples} \
    --keep_ratio ${keep_ratio} \
    --mask_method ${method} \
    --max_seq_length 128  \
    --normal_training ${normal_training} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --max_tokens_per_batch ${max_tokens_per_batch} \
    --add_enc_prefix ${add_enc_prefix} \
    --add_dec_prefix ${add_dec_prefix} \
    --add_cross_prefix ${add_cross_prefix} \
    --prefix_len ${prefix_len} \
    --mid_dim ${mid_dim} \
    --tune_bias ${tune_bias} \
    --add_lora ${add_lora} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --drop_first_layers ${drop_first_layers} \
    --drop_first_adapter_layers ${drop_first_adapter_layers} \
    --drop_first_prefix_layers_enc ${drop_first_prefix_layers_enc} \
    --drop_first_prefix_layers_dec ${drop_first_prefix_layers_dec} \
    --drop_first_prefix_layers_cross ${drop_first_prefix_layers_cross} \
    --add_adapter_gate ${add_adapter_gate} \
    --add_prefix_gate ${add_prefix_gate} \
    --add_lora_gate ${add_lora_gate} \
    --add_central_gate ${add_central_gate} \
    --train_adapter ${train_adapter} \
    --early_stopping_patience ${early_stopping_patience} \
    --seed ${seed} \
    --max_eval_samples ${max_eval_samples} \
    --gradient_accumulation_steps ${gradient_steps} \
    --max_steps ${max_steps} \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${lr} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --max_grad_norm ${max_grad_norm} \
    --weight_decay ${weight_decay} \
    --warmup_steps ${warmup_updates} \
    --warmup_ratio ${warmup_ratio} \
    --fp16 \
    --max_seq_length ${max_seq_length} \
    --logging_steps ${logging_steps} \
    --save_total_limit 2 \
    --evaluation_strategy ${eval_strategy} \
    --save_strategy ${eval_strategy} \
    --save_steps ${save_steps} \
    --eval_steps ${save_steps} \
    --load_best_model_at_end \
    --report_to ${report_to} \
    --run_name ${TASK_NAME}.${DATE}.${exp_name} \
    --overwrite_output_dir \
    --disable_tqdm "True" \
    --metric_for_best_model ${metric} \
    --greater_is_better "True" \
    --ddp_find_unused_parameter "False" \
    --output_dir ${SAVE} ${extra_cmd} \
        2>&1 | tee ${SAVE}/log.txt


## Todo: comment fp16 in for gpu usage
## no     --fp16 \

# persist results on cloud bucket
#echo "Now we start saving"
#echo $PWD
#gsutil cp -r ./checkpoints gs://omega-portal-383613-param-efficient-fine-tuning/checkpoints
# done
