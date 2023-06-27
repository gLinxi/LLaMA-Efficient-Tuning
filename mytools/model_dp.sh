#!/bin/bash

# params
gpu_id='3'
port='6103'
model='/rec_reason_4bit_v1/hf'

# default
model_base_dir="/data/jupyterlab/gzx/LocalModelHub"
# model_base_dir="/home/apps/gzx/LocalModelHub"

python scripts/openai_server_demo/openai_api_server_port.py \
 --base_model ${model_base_dir}${model} \
 --gpus ${gpu_id} \
 --port ${port}
