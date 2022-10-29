#!/bin/bash
#SBATCH --job-name=zh-pixel-s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24 --mem=240000M
#SBATCH -p gpu --gres=gpu:4
#SBATCH --time=2-00:00:00

echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES

export PYTHONPATH=/ceph/hpc/home/euwenyanl/pixel
export WANDB_API_KEY=7ff3068098020a220faf94a829699f6197bb1128
export WANDB_PROJECT=pixel_zh​
export MODEL_NAME="pixel_small_zh"


torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    run_pretraining_pixel_zh.py \
    --text_renderer_name_or_path="/ceph/hpc/home/euwenyanl/pixel/configs/renderers/noto_renderer" \
    --data_dir="/ceph/hpc/home/euwenyanl/storage/wenyan/pixel-zh/preprocessed/" \
    --output_dir="/ceph/hpc/home/euwenyanl/storage/wenyan/pixel-zh/output_b96" \
    --remove_unused_columns=False --label_names=pixel_values \
    --mask_ratio=0.25 --do_train --do_eval --max_steps=500000 \
    --base_learning_rate=1.5e-4 --lr_scheduler_type=cosine \
    --weight_decay=0.05 --num_train_epochs=10 --warmup_ratio=0.05 \
    --per_device_train_batch_size=96 --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=8 --logging_strategy=steps \
    --logging_steps=500 --evaluation_strategy=steps \
    --eval_steps=1000 --save_strategy=steps --save_steps=10000 \
    --seed=42 \
    --masking_max_span_length=3 \
    --half_precision_backend=amp \
    --span_masking --masking_cumulative_span_weights=0.4,0.8,1.0 \
    --use_auth_token=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 \
    --hub_model_id=lyan62/pixel-zh-small \
    --push_to_hub \
    --report_to=wandb \
    --hub_token=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 \
    --remove_unused_columns=False \
    --dropout_prob=0.1    

    