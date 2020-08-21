#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"
export ENRO_DIR=${PWD}/kw_full
export MAX_LEN=100
export BS=8
export GAS=8

python finetune.py \
    --learning_rate=3e-6 \
    --fp16 \
    --do_train \
    --do_predict \
    --val_check_interval=0.25 \
    --adam_eps 1e-06 \
    --num_train_epochs 6 --src_lang ja_XX --tgt_lang ja_XX \
    --data_dir $ENRO_DIR \
    --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
    --train_batch_size=$BS --eval_batch_size=$BS --gradient_accumulation_steps=$GAS \
    --task translation \
    --warmup_steps 500 \
    --freeze_embeds \
    --early_stopping_patience 4 \
    --model_name_or_path=facebook/mbart-large-cc25 \
    --gpus 1 \
    --output_dir jaja_finetune_full \
    --label_smoothing 0.1 \
    --fp16_opt_level=O1 \
    --sortish_sampler \
    $@
