matryoshka_vis_token_scale = 1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

echo 'CHUNKS NUM:' $CHUNKS
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python eval_llava_multiplechoice.py --path_qa_pair_csv ./data/multiple_choice_qa/NExT_QA.csv --path_video /data/NExTVideo_all/%s.mp4 --path_result ./result_nextqa-$matryoshka_vis_token_scale/ --llm_size 7b --matryoshka_vis_token_scale=$matryoshka_vis_token_scale \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX &
done

wait

