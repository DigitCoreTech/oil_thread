import numpy as np
import pandas as pd
import pytest

from service.minimal_curve import compute_trace, compute_circus

_RAD_90 = 0.5 * np.pi


@pytest.fixture
def vertical_trace_df():
    return pd.DataFrame({
        "MD": [0.0, 110.0, 330.0],
        "AZI": [0.0, 0.0, 0.0],
        "INC": [0.0, 00.0, 0.0]
    })


@pytest.fixture
def vertical_computed_trace_df():
    return pd.DataFrame({
        'X': [0.0, 0.0, 0.0],
        'Y': [0.0, 0.0, 0.0],
        'Z': [0.0, 110.0, 330.0],
        'Radius': [np.nan, np.inf, np.inf],
        'tangentX': [0.0, 0.0, 0.0],
        'tangentY': [0.0, 0.0, 0.0],
        'tangentZ': [1.0, 1.0, 1.0],
    })


@pytest.fixture
def circles3_trace_df():
    """
    trace contain 3 circles.
    1st to deep-east(YZ) (Center in (0.0, 200.0, 0.0), R is 200.0).
    2nd to east-north(YX) (Center in (100.0, 200.0, 200.0), R is 100.0).
    last to north-east-deep(XZ) (Center in (100.0, 300.0 + 25.0*2**0.5, 200.0 + 50.0), R is 100.0)
    A = (100.0, 300.0, 200.0)
    :return:
    """
    return pd.DataFrame({
        "MD": [0.0, 200 * _RAD_90, 300 * _RAD_90, 400 * _RAD_90],
        "AZI": [0.0, 90.0, 0.0, 90.0],
        "INC": [0.0, 90.0, 90.0, 45.0]
    })


@pytest.fixture
def circles3_computed_trace_df():
    return pd.DataFrame({
        'X': [0.0, 0.0, 100.0, 200.0],
        'Y': [0.0, 200.0, 300.0, 370.710678],
        'Z': [0.0, 200.0, 200.0, 270.710678],
        'Radius': [np.nan, 200.0, 100.0, 100.0],
        'tangentX': [0.0, 0.0, 1.0, 0.0],
        'tangentY': [0.0, 1.0, 0.0, 0.707107],
        'tangentZ': [1.0, 0.0, 0.0, 0.707107],
    })


def test_vertical_trace(vertical_trace_df: pd.DataFrame):
    computed_df = compute_trace(vertical_trace_df, deg2rad=True, min_angel=0.5)
    # coordinates
    pd.testing.assert_series_equal(computed_df['X'].squeeze().reset_index(drop=True),
                                   pd.Series([0.0, 0.0, 0.0], name='X'))
    pd.testing.assert_series_equal(computed_df['Y'].squeeze().reset_index(drop=True),
                                   pd.Series([0.0, 0.0, 0.0], name='Y'))
    pd.testing.assert_series_equal(computed_df['Z'].squeeze().reset_index(drop=True),
                                   pd.Series([0.0, 110.0, 330.0], name='Z'))
    # tangent
    pd.testing.assert_series_equal(computed_df['tangentX'].squeeze().reset_index(drop=True),
                                   pd.Series([0.0, 0.0, 0.0], name='tangentX'))
    pd.testing.assert_series_equal(computed_df['tangentY'].squeeze().reset_index(drop=True),
                                   pd.Series([0.0, 0.0, 0.0], name='tangentY'))
    pd.testing.assert_series_equal(computed_df['tangentZ'].squeeze().reset_index(drop=True),
                                   pd.Series([1.0, 1.0, 1.0], name='tangentZ'))
    #
    pd.testing.assert_series_equal(computed_df['gamma'].squeeze().reset_index(drop=True),
                                   pd.Series([np.nan, 0.0, 0.0], name='gamma'))
    pd.testing.assert_series_equal(computed_df['Radius'].squeeze().reset_index(drop=True),
                                   pd.Series([np.nan, np.inf, np.inf], name='Radius'))
    assert len(vertical_trace_df) == 3


def test_circles3_trace(circles3_trace_df: pd.DataFrame):
    computed_df = compute_trace(circles3_trace_df, deg2rad=True, min_angel=0.5)
    # coordinates
    pd.testing.assert_series_equal(computed_df['X'].squeeze().reset_index(drop=True),
                                   pd.Series([0.0, 0.0, 100.0, 200], name='X'))
    pd.testing.assert_series_equal(computed_df['Y'].squeeze().reset_index(drop=True),
                                   pd.Series([0.0, 200.0, 300.0, 370.710678], name='Y'))
    pd.testing.assert_series_equal(computed_df['Z'].squeeze().reset_index(drop=True),
                                   pd.Series([0.0, 200.0, 200.0, 270.710678], name='Z'))
    #
    pd.testing.assert_series_equal(computed_df['gamma'].squeeze().reset_index(drop=True),
                                   pd.Series([np.nan, _RAD_90, _RAD_90, _RAD_90], name='gamma'))
    pd.testing.assert_series_equal(computed_df['Radius'].squeeze().reset_index(drop=True),
                                   pd.Series([np.nan, 200.0, 100.0, 100.0], name='Radius'))
    assert len(computed_df) == 4


def test_vertical_trace_circus(vertical_computed_trace_df: pd.DataFrame):
    # before
    nan_vector = [np.nan, np.nan, np.nan]
    # do
    circus_df = compute_circus(vertical_computed_trace_df)
    # then
    pd.testing.assert_series_equal(circus_df['C_X'].squeeze().reset_index(drop=True),
                                   pd.Series(nan_vector, name='C_X'))
    pd.testing.assert_series_equal(circus_df['C_Y'].squeeze().reset_index(drop=True),
                                   pd.Series(nan_vector, name='C_Y'))
    pd.testing.assert_series_equal(circus_df['C_Z'].squeeze().reset_index(drop=True),
                                   pd.Series(nan_vector, name='C_Z'))
    assert len(circus_df) == 3


def test_circles3_trace_circus(circles3_computed_trace_df: pd.DataFrame):
    # do
    circus_df = compute_circus(circles3_computed_trace_df)
    # then
    pd.testing.assert_series_equal(circus_df['C_X'].squeeze().reset_index(drop=True),
                                   pd.Series([np.nan, 0.0, 100.0, 100.0], name='C_X'))
    pd.testing.assert_series_equal(circus_df['C_Y'].squeeze().reset_index(drop=True),
                                   pd.Series([np.nan, 200.0, 200.0, 370.710678], name='C_Y'))
    pd.testing.assert_series_equal(circus_df['C_Z'].squeeze().reset_index(drop=True),
                                   pd.Series([np.nan, 0.0, 200.0, 270.710678], name='C_Z'))

    assert len(circus_df) == 4
