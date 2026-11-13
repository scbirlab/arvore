#!/usr/bin/env bash

set -euox pipefail

TRAIN="hf://scbirlab/thomas-2018-spark-wt~Acinetobacter-baumannii:train"
TEST="hf://scbirlab/thomas-2018-spark-wt~Acinetobacter-baumannii:validation"
LLM="transformer://scbirlab/lchemme-base-zinc22-lteq300:clean_smiles~mean"

script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs/strain
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
                "mic_method_hash": ["mic_method", "hash"],
                "mic_ineq_operator_hash": ["mic_ineq_operator", "hash"],
                "species_vec": ["full_strain_name", "vectome-fingerprint"]
            }
        },
        "features": ["fp", "mic_method_hash", "mic_ineq_operator_hash", "species_vec"],
        "target": "pmic"
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
 