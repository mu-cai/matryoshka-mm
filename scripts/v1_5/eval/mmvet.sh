#!/bin/bash
matryoshka_vis_token_scale=$1
CKPT=llava-v1.5-7b-m3-${matryoshka_vis_token_scale}
python -m llava.eval.model_vqa \
    --model-path mucai/llava-v1.5-7b-m3 \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$CKPT.jsonl \
    --temperature 0 \
    --matryoshka_vis_token_scale $matryoshka_vis_token_scale \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$CKPT.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$CKPT.json

