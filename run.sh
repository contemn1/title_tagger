if [ ${PREVIOUS_JOB_OUTPUT_DIR} ]; then
    python train.py --training-dir "${DATA_DIR}"  \
    --training-file "result_20170321_20171231_filtered_v2.txt" \
    --batch-size 128 \
    --rnn-size 64 \
    --word-vec-size 64 \
    --model-path "${OUTPUT_DIR}" \
    --num-epoches 1000 \
    --run-valid-every 10000 --save-model-every 20000 \
    --early-stop-tolerance 45 \
    --print-loss-every 200 --normalize-attention True \
    --dropout 0.5 --word-index-map-name "word_to_index_v2.txt" \
    --tag-index-map-name "tag_to_index_v2.txt" --store-dict False \
    --previous-output-dir ${PREVIOUS_JOB_OUTPUT_DIR} \
    --restore-model True

else
    python train.py --training-dir "${DATA_DIR}"  \
    --training-file "result_20170321_20171231_filtered_v3.txt" \
    --batch-size 128 \
    --rnn-size 64 \
    --word-vec-size 64 \
    --model-path "${OUTPUT_DIR}" \
    --num-epoches 1000 \
    --run-valid-every 10000 --save-model-every 20000 \
    --early-stop-tolerance 45 \
    --print-loss-every 200 --normalize-attention True \
    --dropout 0.5 --word-index-map-name "word_to_index_v3.txt" \
    --tag-index-map-name "tag_to_index_v3.txt" --store-dict True \
fi
