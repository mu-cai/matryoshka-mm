#!/bin/bash
matryoshka_vis_token_scale=$1
CKPT=llava-v1.5-7b-m3-${matryoshka_vis_token_scale}
if [ "$1" = "dev" ]; then
    ZH_SPLIT="验证集"
    echo "Evaluating in 'dev' split."
elif [ "$1" = "test" ]; then
    ZH_SPLIT="测试集"
    echo "Evaluating in 'test' split."
else
    echo "Unknown split, please choose between 'dev' and 'test'."
    exit 1
fi

python -m llava.eval.model_vqa_qbench \
    --model-path mucai/llava-v1.5-7b-m3 \
    --image-folder ./playground/data/eval/qbench/images_llvisionqa/ \
    --questions-file ./playground/data/eval/qbench/质衡-问答-$ZH_SPLIT.json \
    --answers-file ./playground/data/eval/qbench/llvisionqa_zh_$1_answers.jsonl \
    --conv-mode llava_v1 \
    --lang zh
