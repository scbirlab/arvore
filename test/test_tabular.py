"""Column matrices from Datasets and column mappings."""

import numpy as np
import pytest

from arvore.tabular import matrix

def test_matrix_groups_columns():
    data = {"a": [1., 2.], "b": [[3., 4.], [5., 6.]], "y": [7., 8.]}

    X, y = matrix(data, (("a", "b"), ("y",)))

    np.testing.assert_array_equal(X, [[1., 3., 4.], [2., 5., 6.]])
    np.testing.assert_array_equal(y, [[7.], [8.]])


def test_matrix_counts_rows_not_columns():
    with pytest.raises(ValueError, match="No rows"):
        matrix({"a": [], "b": []}, (("a",),))

    (X,) = matrix({"a": [1., 2., 3.]}, (("a",),))
    assert X.shape == (3, 1)


def test_matrix_accepts_boolean_columns():
    data = {"has_ortholog": [True, False, True], "fp": [[1, 0], [0, 1], [1, 1]]}

    (X,) = matrix(data, (("has_ortholog", "fp"),))

    np.testing.assert_array_equal(X, [[1, 1, 0], [0, 0, 1], [1, 1, 1]])


def test_matrix_nulls_become_nan():
    data = {"flag": [True, None, False], "y": [1.5, 2.5, None]}

    X, y = matrix(data, (("flag",), ("y",)))

    np.testing.assert_array_equal(X, [[1.], [np.nan], [0.]])
    np.testing.assert_array_equal(y, [[1.5], [2.5], [np.nan]])


def test_matrix_rejects_non_numeric_columns():
    with pytest.raises(TypeError, match="must be numeric"):
        matrix({"s": ["a", "b"]}, (("s",),))
    with pytest.raises(TypeError, match="must be numeric"):
        matrix({"s": ["1", None]}, (("s",),))
