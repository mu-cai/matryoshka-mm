#!/bin/bash
matryoshka_vis_token_scale=$1
CKPT=llava-v1.5-7b-m3-${matryoshka_vis_token_scale}
python -m llava.eval.model_vqa_loader \
    --model-path mucai/llava-v1.5-7b-m3 \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${CKPT}.jsonl \
    --temperature 0 \
    --matryoshka_vis_token_scale $matryoshka_vis_token_scale \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${CKPT}.jsonl
