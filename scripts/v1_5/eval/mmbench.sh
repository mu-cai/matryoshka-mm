#!/bin/bash
matryoshka_vis_token_scale=$1
CKPT=llava-v1.5-7b-m3-${matryoshka_vis_token_scale}
SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path mucai/llava-v1.5-7b-m3 \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --matryoshka_vis_token_scale $matryoshka_vis_token_scale \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT
