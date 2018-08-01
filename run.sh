python train.py --training-dir "${DATA_DIR}"  \
--training-file "filtered_sorted_result.txt" \
--batch-size 64 \
--rnn-size 64 \
--word-vec-size 64 \
--model-path "${OUTPUT_DIR}" \
--num-epoches 1000 \
--run-valid-every 20000 --save-model-every 40000