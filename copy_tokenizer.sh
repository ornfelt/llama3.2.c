#!/usr/bin/env bash

# Example usage:
# ./copy_tokenizer.sh
# ./copy_tokenizer.sh /some/other/model/path

MODEL_DIR="${1:-/media/my_filesmy_docs/ai/models/small_models/Llama3.2-1B-Instruct}"
DATA_DIR="$(pwd)/data"

if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory does not exist: $DATA_DIR"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "Model directory does not exist: $MODEL_DIR"
    exit 1
fi

TOKENIZER_JSON="$MODEL_DIR/tokenizer.json"
TOKENIZER_MODEL="$MODEL_DIR/tokenizer.model"

if [ ! -f "$TOKENIZER_JSON" ]; then
    echo "Missing file: $TOKENIZER_JSON"
    exit 1
fi

if [ ! -f "$TOKENIZER_MODEL" ]; then
    echo "Missing file: $TOKENIZER_MODEL"
    exit 1
fi

cp "$TOKENIZER_JSON" "$DATA_DIR/tokenizer.json"
cp "$TOKENIZER_MODEL" "$DATA_DIR/llama3_tokenizer.model"

echo "Copied files to: $DATA_DIR"
