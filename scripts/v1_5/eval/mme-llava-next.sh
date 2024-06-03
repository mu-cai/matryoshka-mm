#!/bin/bash
matryoshka_vis_token_scale=$1
CKPT=llava-next-vicuna-7b-m3-${matryoshka_vis_token_scale}

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-next-vicuna-7b-m3 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$CKPT.jsonl \
    --temperature 0 \
    --matryoshka_vis_token_scale $matryoshka_vis_token_scale \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
