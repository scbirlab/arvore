import json

import pytest
import yaml

from arvore.config.instantiate import apply_overrides


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_file_overrides(tmp_path, suffix):
    path = tmp_path / f"overrides{suffix}"
    values = {
        "box.model.init_args.n_estimators": 500,
        "box.model.init_args.bootstrap": False,
        "fit.validation": None,
        "fit.metrics": ["rmse", "mae"],
    }

    if suffix == ".json":
        path.write_text(json.dumps(values))
    else:
        path.write_text(yaml.safe_dump(values))

    result = apply_overrides({}, [str(path)])

    assert result == {
        "box": {
            "model": {
                "init_args": {
                    "n_estimators": 500,
                    "bootstrap": False,
                }
            }
        },
        "fit": {
            "validation": None,
            "metrics": ["rmse", "mae"],
        },
    }


def test_order_and_inline_compatibility(tmp_path):
    path = tmp_path / "overrides.yaml"
    path.write_text(
        "model.n_estimators: 500\n"
        "model.bootstrap: false\n"
    )

    result = apply_overrides(
        {"model": {"n_estimators": 100}},
        [
            "model.n_estimators=200",
            str(path),
            "model.bootstrap=true",
            "model.name=rf",
        ],
    )

    assert result == {
        "model": {
            "n_estimators": 500,
            "bootstrap": True,
            "name": "rf",
        },
    }


def test_does_not_mutate_input(tmp_path):
    config = {"model": {"n_estimators": 100}}
    path = tmp_path / "overrides.json"
    path.write_text('{"model.n_estimators": 500}')

    apply_overrides(config, [str(path)])

    assert config["model"]["n_estimators"] == 100


@pytest.mark.parametrize(
    "contents",
    ["[1, 2, 3]", "null", "42"],
)
def test_rejects_non_mapping(tmp_path, contents):
    path = tmp_path / "overrides.yaml"
    path.write_text(contents)

    with pytest.raises(ValueError, match="must contain a mapping"):
        apply_overrides({}, [str(path)])


@pytest.mark.parametrize(
    "key",
    ["", ".model", "model.", "model..depth"],
)
def test_rejects_invalid_keys(tmp_path, key):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({key: 100}))

    with pytest.raises(ValueError, match="Invalid config override path"):
        apply_overrides({}, [str(path)])


def test_rejects_non_string_key(tmp_path):
    path = tmp_path / "overrides.yaml"
    path.write_text("123: value\n")

    with pytest.raises(ValueError, match="must be strings"):
        apply_overrides({}, [str(path)])


def test_rejects_scalar_intermediate(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text('{"model.depth": 10}')

    with pytest.raises(ValueError, match="not a mapping"):
        apply_overrides({"model": "rf"}, [str(path)])


def test_missing_file(tmp_path):
    path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        apply_overrides({}, [str(path)])


def test_filename_containing_equals(tmp_path):
    path = tmp_path / "experiment=v2.yaml"
    path.write_text("model.depth: 10\n")

    assert apply_overrides({}, [str(path)]) == {
        "model": {"depth": 10}
    }
