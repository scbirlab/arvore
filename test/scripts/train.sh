#!/usr/bin/env bash
set -euox pipefail

TRAIN="hf://scbirlab/fang-2023-biogen-adme~scaffold-split:train"
TEST="hf://scbirlab/fang-2023-biogen-adme~scaffold-split:test"
LLM="transformer://scbirlab/lchemme-base-zinc22-lteq300:clean_smiles~mean"

script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs/original
CACHE="$OUTPUT_DIR/cache"
OUTPUT="$OUTPUT_DIR/models"
CONFIG="$OUTPUT_DIR/config.json"

START=50
STOP=550

mkdir -p "$(dirname "$CONFIG")"

printf \
'{
    "box": {
        "model": {
            "class_path": "sklearn.ensemble.RandomForestRegressor",
            "init_args": {
                "n_estimators": 20,
                "random_state": 42
            }
        },
        "pipeline": {
            "column_transforms": {
                "fp": ["smiles", "morgan-fingerprint"],
                "2d": ["smiles", "descriptors-2d"]
            }
        },
        "features": ["clogp", "fp", "2d"],
        "target": "log_rlm"
    }
}' \
> "$CONFIG"

HF_HOME="$CACHE" \
arvore train \
    "$TEST" \
    -2 "$TRAIN" \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT/model.ao" \
    --cache "$CACHE"

HF_HOME="$CACHE" \
arvore predict \
    "$OUTPUT/model.ao" \
    --trust-sklearn-trees \
    --data "$TRAIN" \
    --start "$START" \
    --stop "$STOP" \
    --checkpoint "$OUTPUT/pred.pv" \
    --output "$OUTPUT/pred.csv" \
    --cache "$CACHE"
 