#!/bin/bash
matryoshka_vis_token_scale=$1
CKPT=llava-v1.5-7b-m3-${matryoshka_vis_token_scale}
python -m llava.eval.model_vqa_science \
    --model-path mucai/llava-v1.5-7b-m3 \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --matryoshka_vis_token_scale $matryoshka_vis_token_scale \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}_result.json
