docker run -v ./data:/data  \
    -v ./src:/src \
    -e HUGGING_FACE_HUB_TOKEN="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" \
    ludwigai/ludwig:latest \
    experiment --config /src/config.yaml \
        --dataset /data/train-00000-of-00001.parquet \
        --output_directory /src/results