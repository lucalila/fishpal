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

export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model

cache_dir=${TRANSFORMERS_CACHE}


TASK_NAME=sst2  # todo: second file for mnli
metric="accuracy"

# wandb env variables
export WANDB_PROJECT=glue.${TASK_NAME}
export WANDB_WATCH="false"

DATE=`date +%Y%m%d`

# declare -a root_seed_list=(42 2 4 6 8)    # todo: s = {111, 222, 333, 444, 555} from Unipelt paper
# seed=${root_seed_list[$SLURM_ARRAY_TASK_ID]}

seed=42

# declare -a seed_list=(42)
# declare -a seed_list=(42 2 4)
# declare -a seed_list=(8)
# declare -a seed_list=(6 8)
# declare -a seed_list=(${root_seed})

#----- Houlsby Adapter -----
attn_mode="adapter"  # OK
attn_option="sequential"  # OK
attn_composition="add"  # OK
attn_bn=200  # attn bottleneck dim

ffn_mode="none"  # OK
ffn_option="none"  # OK
ffn_adapter_layernorm_option="none"  # OK
ffn_adapter_init_option="bert"  # OK
ffn_adapter_scalar="1"  # OK
ffn_bn=-1 # ffn bottleneck dim  # OK

#---- prefix tuning baseline -----
attn_mode="prefix"
attn_option="concat"
attn_composition="add"
attn_bn=200  # attn bottleneck dim

ffn_mode="none"
ffn_option="parallel"
ffn_adapter_layernorm_option="none"
ffn_adapter_init_option="lora"  # todo: from the paper, they use the lora init as in the orig (Li & Liang)
ffn_adapter_scalar="4"
ffn_bn=512 # ffn bottleneck dim



#----- lora -----
attn_mode="lora"
attn_option="none"
attn_composition="add"
attn_bn=16

# set ffn_mode to be 'lora' to use
# lora at ffn as well

ffn_mode="none"  # todo: setting this to lora would be an ablation of He et al. not the orig version
ffn_option="none"
ffn_adapter_layernorm_option="none"
ffn_adapter_init_option="bert"
ffn_adapter_scalar="1"
ffn_bn=16

# lora params are not set
if [ -z ${lora_alpha+x} ];
then
    lora_alpha=0
    lora_init="lora"
    lora_dropout=0
fi

# set to 1 for debug mode which only
# uses 1600 training examples
debug=0  # OK

# set to "wandb" to use weights & bias
report_to="none" # this is just a visualization tool

bsz=2  # ADJUSTED (from He et al., sents -> automatically sents? Houlsby also uses 32 batch size)
gradient_steps=1 # todo: what does it mean?

lr=1e-4 # todo: lr is from He et al., Pfeiffer: 1e-4; Houlsby uses: 3e-4
max_grad_norm=1  # OK (this is the default from huggingface)
weight_decay=0.1  # OK (this is the RoBERTa default)
warmup_updates=0  # OK (this is the default)
warmup_ratio=0.06  # OK (this is from He et al., default is 0.0)
max_steps=-1  # OK (this is the default from huggingface and overwritten from epochs and batch size)
num_train_epochs=10 # OK (from He et al., Houlsby: num_train_epochs=5.0)
max_tokens_per_batch=0  # todo: which value?
max_seq_length=512 # ADJUSTED (from He et al.)
#max_train_samples=100 # ADJUSTED -> 500, 1000, all # K = {100, 500, 1000 , all} from paper, all has the best performance (from unipelt)

lr_scheduler_type="polynomial"  # OK (this is from He et al.)
unfreeze='ef_'  # todo: which value?
max_eval_samples=1600 # OK
logging_steps=50 # OK

eval_strategy="epoch" # OK
save_steps=5000 # OK

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



# for seed in "${seed_list[@]}"; do

exp_name=glue.${TASK_NAME}.am_${attn_mode}.ao_${attn_option}.fm_${ffn_mode}
exp_name+=.fo_${ffn_option}.abn${preseqlen}.fbn${ffn_bn_len}.ac_${attn_composition}
exp_name+=.fl_${ffn_adapter_layernorm_option}.finit_${ffn_adapter_init_option}
exp_name+=.fs_${ffn_adapter_scalar}.unfrz_${unfreeze}.ne${num_train_epochs}
exp_name+=.warm${warmup_ratio}.wd${weight_decay}.seed${seed}.${debug_str}
SAVE=checkpoints/glue/${TASK_NAME}/${DATE}/${exp_name}
echo "${SAVE}"
rm -rf ${SAVE}; mkdir -p ${SAVE}

rm checkpoints/hf_model/downloads/*.lock
rm checkpoints/hf_model/*.lock


# python -m torch.distributed.launch --nproc_per_node 2 --master_port=${port} examples/pytorch/text-classification/run_glue.py \

python -u examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path roberta-base \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --max_tokens_per_batch ${max_tokens_per_batch} \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --lora_init ${lora_init} \
    --attn_mode ${attn_mode} \
    --attn_option ${attn_option} \
    --attn_composition ${attn_composition} \
    --ffn_mode ${ffn_mode} \
    --ffn_option ${ffn_option} \
    --ffn_adapter_layernorm_option ${ffn_adapter_layernorm_option} \
    --ffn_adapter_scalar ${ffn_adapter_scalar} \
    --ffn_adapter_init_option ${ffn_adapter_init_option} \
    --mid_dim 800 \
    --attn_bn ${attn_bn} \
    --ffn_bn ${ffn_bn} \
    --seed ${seed} \
    --unfreeze_params ${unfreeze} \
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
    --max_seq_length ${max_seq_length} \
    --fp16 \
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

# persist results on cloud bucket
#echo "Now we start saving"
#echo $PWD
#gsutil cp -r ./checkpoints gs://omega-portal-383613-param-efficient-fine-tuning/checkpoints
# done