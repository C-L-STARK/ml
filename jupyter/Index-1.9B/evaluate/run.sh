python run.py --datasets hellaswag_ppl \
    --hf-path ./models/index_1_9B \
    --model-kwargs device_map='auto' trust_remote_code=True \
    --tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True \
    --max-out-len 1 \
    --max-seq-len 4096 \
    --batch-size 64 \
    --no-batch-padding \
    --num-gpus 1
