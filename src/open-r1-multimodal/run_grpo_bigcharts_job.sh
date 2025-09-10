export HF_HOME=/mnt/tmp/hf_cache

cd /mnt/home/rl_exp/VLM-R1/src/open-r1-multimodal

conda init
. ~/.bashrc
source ~/.bashrc
conda activate /mnt/home/.conda/envs/vlm-r1
/mnt/home/.conda/envs/vlm-r1/bin/wandb login 0c07e7d1f954810b6028d2e41d573a5190643e88

export DEBUG_MODE="true"
RUN_NAME="Qwen2.5-VL-3B-GRPO-All-RL-Test"

export LOG_PATH="/mnt/tmp/rl_data/outputs/bigcharts_all/Qwen2.5-VL-3B-GRPO-All-RL-Test/debug_log_reasoning_$RUN_NAME.txt"

/mnt/home/.conda/envs/vlm-r1/bin/torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_bigcharts.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir /mnt/tmp/rl_data/outputs/bigcharts_all/Qwen2.5-VL-3B-GRPO-All-RL-Test/ \
    --model_name_or_path /mnt/tmp/rl_data/outputs/bigcharts_all/qwen2_5_vl-3b_sft_cot_2/checkpoint-21599-Qwen2.5-VL/ \
    --dataset_name data_config/rl_data.yaml \
    --image_root "/mnt/tmp/rl_data/bigmathvis/compiled_datasets/rl_data_small/images/" \
    --reward_funcs "cerm" "format" \
    --max_prompt_length 1024 \
    --max-completion-length 2048 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true