#!/usr/bin/env bash
set -euox pipefail

script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs/chk-config-reload

bash "$script_dir"/train-scorers.sh "$OUTPUT_DIR"

workdir="$OUTPUT_DIR"

arvore train "$workdir/train.csv" \
    --config "$workdir/model.ao" \
    --cache "$workdir/cache" \
    --checkpoint "$workdir/retrained.ao" \
    --set box.model.base.init_args.n_estimators=17

python -c '
import json
from pathlib import Path

path = Path("'"$OUTPUT_DIR"'/retrained.ao/config.json")
manifest = json.loads(path.read_text())

model = manifest["config"]["model"]
assert model["base"]["init_args"]["n_estimators"] == 17

assert "model_config" not in manifest["config"]
assert manifest["config"]["features"] == ["x"]
assert manifest["config"]["target"] == ["y"]

print("PASS: trained a fresh model from checkpoint configuration")
'
