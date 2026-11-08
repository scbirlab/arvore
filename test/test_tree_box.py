import json
from argparse import Namespace

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from treeton import Box
from treeton.cli.train import _train
from treeton.cli.predict import _predict


def test_tree_box_roundtrip(tmp_path):
    train = tmp_path / 'training.parquet'
    test = tmp_path / 'test.parquet'
    x = np.linspace(0, 3, 24)
    pd.DataFrame({'x': x, 'y': 2 * x + 1}).to_parquet(train)
    pd.DataFrame({'x': [0.7, 1.5, 2.8]}).to_parquet(test)

    config = {
        'box': {
            'model': {'class_path': 'sklearn.ensemble.RandomForestRegressor',
                      'init_args': {'n_estimators': 12, 'random_state': 1}},
            'pipeline': {},
            'features': ['x'], 'target': 'y',
        },
    }
    config_file = tmp_path / 'experiment.json'
    config_file.write_text(json.dumps(config))
    checkpoint = tmp_path / 'model.tt'
    output = tmp_path / 'predictions.parquet'

    _train(Namespace(config=str(config_file), model=None, set=None,
                     training=str(train), validation=None,
                     cache=str(tmp_path / 'cache'), output=str(checkpoint)))
    assert (checkpoint / 'data' / 'config.json').exists()

    loaded = Box.load(checkpoint, cache_dir=str(tmp_path / 'cache'))
    expected = loaded.predict(str(test))['prediction']
    _predict(Namespace(checkpoint=str(checkpoint), data=str(test),
                       output=str(output), config=None, set=None,
                       cache=str(tmp_path / 'cache')))
    assert np.allclose(pd.read_parquet(output)['prediction'], expected)
    assert np.isfinite(np.asarray(expected, dtype=float)).all()


def test_logical_aspect_feature_alias(tmp_path):
    train = tmp_path / 'train.parquet'
    test = tmp_path / 'test.parquet'
    pd.DataFrame({'x': [1., 2., 3., 4., 5.],
                  'y': [2., 3., 5., 7., 11.]}).to_parquet(train)
    pd.DataFrame({'x': [1.5, 3.5]}).to_parquet(test)
    config = {
        'model': {'class_path': 'sklearn.ensemble.RandomForestRegressor',
                  'init_args': {'n_estimators': 5, 'random_state': 0}},
        'pipeline': {'column_transforms': {'log_x': ['x', 'log']}},
        'features': ['log_x'], 'target': 'y',
    }
    box = Box.from_config(config)
    box.fit(str(train))
    before = np.asarray(box.predict(str(test))['prediction'])
    path = tmp_path / 'model.tt'
    box.save(path)
    after = np.asarray(Box.load(path).predict(str(test))['prediction'])
    np.testing.assert_allclose(before, after)
