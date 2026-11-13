
set -euox pipefail
# Prepare the smoke-test dataset and config.
script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs/no-refs

bash "$script_dir"/train-scorers.sh "$OUTPUT_DIR"

workdir="$OUTPUT_DIR"

treeton train "$workdir/train.csv" \
    --config "$workdir/experiment.json" \
    --cache "$workdir/cache" \
    --checkpoint "$workdir/model-no-refs.tt" \
    --skip-ref-save

python -c '
import json
from pathlib import Path

path = Path("'"$OUTPUT_DIR"'/model-no-refs.tt")
manifest = json.loads((path / "config.json").read_text())

assert "references" not in manifest["artefacts"]
assert not (path / "references").exists()
assert (path / manifest["artefacts"]["training_data"]).exists()

print("PASS: CLI omitted references but retained training data")
'

treeton predict "$workdir/model-no-refs.tt" \
    --trust-sklearn-trees \
    --cache "$workdir/cache" \
    --data "$workdir/candidates.csv" \
    --score 'feature(n_neighbors=2,return_var=True)' \
    --output "$workdir/predictions-no-refs.csv"

python -c '
import csv
from pathlib import Path

path = Path("'"$OUTPUT_DIR"'/predictions-no-refs.csv")

with path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)

assert len(rows) == 4
assert "score:feature" in reader.fieldnames
assert "score:feature:var" in reader.fieldnames

print("PASS: reference scoring works after loading an opt-out checkpoint")
'
