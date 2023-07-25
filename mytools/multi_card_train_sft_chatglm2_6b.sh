#!/bin/bash

# date
date=`date +%Y%m%d%H`

# params
model="intention_4bit_v1_chatglm26b"
sft_data="intention_train"
output_dir="./model/"${model}"_ckp"
log_dir="./log/"${model}"_"${date}

# default
model_base_dir="/home/apps/gzx/LocalModelHub"
# model_base_dir="/data/jupyterlab/gzx/LocalModelHub/"
chatglm2_6b=${model_base_dir}"/chatglm2_6b/hf"
baichuan_7b=${model_base_dir}"/baichuan/hf"

set -x
accelerate launch ./src/train_sft.py \
    --model_name_or_path ${baichuan_7b} \
    --do_train \
    --dataset  ${sft_data}\
    --finetuning_type lora \
    --output_dir ${output_dir} \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --learning_rate 1e-3 \
    --max_source_length 2048 \
    --max_target_length 2048 \
    --ddp_find_unused_parameters False \
    --num_train_epochs 3.0 \
    --quantization_bit 4 \
    --fp16 &>> ${log_dir}
