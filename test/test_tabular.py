"""Column matrices from Datasets and column mappings."""

from types import SimpleNamespace

import numpy as np
import pytest

from arvore.tabular import matrix

NO_TRANSFORMS = SimpleNamespace(column_transforms={})


def test_matrix_groups_columns():
    data = {"a": [1., 2.], "b": [[3., 4.], [5., 6.]], "y": [7., 8.]}

    X, y = matrix(data, NO_TRANSFORMS, (("a", "b"), ("y",)))

    np.testing.assert_array_equal(X, [[1., 3., 4.], [2., 5., 6.]])
    np.testing.assert_array_equal(y, [[7.], [8.]])


def test_matrix_counts_rows_not_columns():
    with pytest.raises(ValueError, match="No rows"):
        matrix({"a": [], "b": []}, NO_TRANSFORMS, (("a",),))

    (X,) = matrix({"a": [1., 2., 3.]}, NO_TRANSFORMS, (("a",),))
    assert X.shape == (3, 1)
