#!/usr/bin/env bash
set -euox pipefail

# End-to-end CLI smoke test for prediction scorers.
#
# Requires:
#   pip install -e ".[dev]"
#
# Run:
#   bash test/smoke_scorers.sh
#
# All work is done in a temporary directory.
# The test calls the installed `treeton` executable directly.

command -v treeton >/dev/null || {
    echo "ERROR: treeton executable not found" >&2
    exit 1
}
script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs/scorers
workdir=${1:-"$OUTPUT_DIR"}
mkdir -p "$workdir"

echo "[1/5] Creating datasets and configuration"

cat > "$workdir/train.csv" <<'CSV'
x,smiles,y
0,CCO,1.1
1,CCN,1.8
2,CCC,3.2
3,CCCl,3.7
4,CCBr,4.9
5,CCF,6.2
6,CCCO,7.0
7,CCCN,7.8
8,CCCC,9.1
9,CCOC,9.8
10,CCCOC,11.1
11,CC(C)O,12.3
12,CC(C)N,12.9
13,CC(=O)O,14.0
14,CC(=O)N,15.3
15,c1ccccc1,15.8
16,Cc1ccccc1,17.1
17,Oc1ccccc1,18.2
18,NCc1ccccc1,19.0
19,CCc1ccccc1,20.1
CSV

cat > "$workdir/candidates.csv" <<'CSV'
x,smiles
2.5,CCO
7.5,CCCO
12.5,CC(=O)O
16.5,c1ccccc1
CSV

cat > "$workdir/experiment.json" <<'JSON'
{
  "box": {
    "model": {
      "kind": "ensemble",
      "base": {
        "class_path": "sklearn.ensemble.RandomForestRegressor",
        "init_args": {
          "n_estimators": 12,
          "max_depth": 5,
          "n_jobs": 1
        }
      },
      "n_members": 2,
      "seed": 42
    },
    "pipeline": {},
    "features": ["x"],
    "target": "y"
  },
  "fit": {
    "skip_eval": true
  },
  "save": {
    "score": {
      "leaf_percentile": {
        "class_path": "treeton.scorers.leaves.InverseLeafOccupancy",
        "init_args": {"normalization": "reference_percentile"}
      },
      "feature_distance": "feature_distance"
    }
  }
}
JSON

cat > "$workdir/predict.json" <<'JSON'
{
  "batch_size": 2,
  "score": {
    "leaf_percentile": {
      "class_path": "treeton.scorers.leaves.InverseLeafOccupancy",
      "init_args": {
        "normalization": "reference_percentile"
      }
    }
  }
}
JSON

echo "[2/5] Training and saving ensemble"

treeton train "$workdir/train.csv" \
    --config "$workdir/experiment.json" \
    --cache "$workdir/cache" \
    --checkpoint "$workdir/model.tt"

test -d "$workdir/model.tt" || {
    echo "ERROR: checkpoint directory missing" >&2
    exit 1
}

echo "[3/5] Predicting with configured and CLI scorers"

treeton predict "$workdir/model.tt" \
    --trust-sklearn-trees \
    --data "$workdir/candidates.csv" \
    --config "$workdir/predict.json" \
    --cache "$workdir/cache" \
    --score ensemble_sd tree_sd inverse_leaf_occupancy feature_distance \
    --disaggregate \
    --output "$workdir/predictions.csv"

test -s "$workdir/predictions.csv" || {
    echo "ERROR: prediction CSV missing or empty" >&2
    exit 1
}

echo "[4/5] Checking prediction and scorer columns"

header="$(head -n 1 "$workdir/predictions.csv")"

for column in \
    "prediction:y:" \
    "score:ensemble_sd:y" \
    "score:tree_sd:y" \
    "score:inverse_leaf_occupancy" \
    "score:feature_distance" \
    "score:leaf_percentile"
do
    if [[ "$header" != *"$column"* ]]; then
        echo "ERROR: missing column: $column" >&2
        echo "Actual header: $header" >&2
        exit 1
    fi
done

rows="$(wc -l < "$workdir/predictions.csv")"
rows="${rows//[[:space:]]/}"

if [[ "$rows" -ne 5 ]]; then
    echo "ERROR: expected 4 predictions, got $((rows - 1))" >&2
    exit 1
fi

echo "[5/5] Testing Tanimoto SMILES fingerprint generation"

# Parquet is used because the generated fingerprint is an
# array-valued column, which is not naturally CSV-compatible.
treeton predict "$workdir/model.tt" \
    --trust-sklearn-trees \
    --cache "$workdir/cache" \
    --data "$workdir/candidates.csv" \
    --score tanimoto_distance \
    --output "$workdir/tanimoto.parquet"

test -s "$workdir/tanimoto.parquet" || {
    echo "ERROR: Tanimoto Parquet output missing" >&2
    exit 1
}

echo
echo "PASS: treeton CLI scorer smoke test"
