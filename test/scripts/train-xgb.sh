#!/usr/bin/env bash
set -euox pipefail

TRAIN="hf://scbirlab/fang-2023-biogen-adme~scaffold-split:train"
TEST="hf://scbirlab/fang-2023-biogen-adme~scaffold-split:test"
LLM="transformer://scbirlab/lchemme-base-zinc22-lteq300:clean_smiles~mean"

script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs/xgb
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
      "class_path": "xgboost.XGBRegressor",
      "init_args": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "tree_method": "hist",
        "eval_metric": "rmse",
        "random_state": 42,
        "n_jobs": 4
      }
    },
    "features": ["fp"],
    "target": "log_rlm",
    "pipeline": {
        "column_transforms": {
            "fp": ["smiles", "morgan-fingerprint"],
            "2d": ["smiles", "descriptors-2d"]
        }
    }
  },
  "fit": {
    "use_validation_for_fit": true,
    "early_stopping_rounds": 30,
    "skip_eval": false
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
    --data "$TRAIN" \
    --start "$START" \
    --stop "$STOP" \
    --checkpoint "$OUTPUT/pred.pv" \
    --output "$OUTPUT/pred.csv" \
    --cache "$CACHE"
 