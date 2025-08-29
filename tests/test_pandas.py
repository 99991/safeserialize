from safeserialize import dumps, loads
import pandas as pd
import numpy as np
import random
from safeserialize.types.numpy import _allowed_dtypes as numpy_dtypes

# Functions to test whether data == loads(dumps(data))
# for pd.Series, pd.DataFrame and pd.Index
def roundtrip_series(s):
    serialized_data = dumps(s)
    deserialized_series = loads(serialized_data)
    pd.testing.assert_series_equal(s, deserialized_series)

def roundtrip_df(df):
    serialized_data = dumps(df)
    deserialized_df = loads(serialized_data)
    pd.testing.assert_frame_equal(df, deserialized_df)

def roundtrip_index(index):
    serialized_data = dumps(index)
    deserialized_index = loads(serialized_data)
    pd.testing.assert_index_equal(index, deserialized_index)

def test_pandas():
    # Test various data types
    a = pd.Series([1, 2, None, 4], dtype="Int64", name="int_nullable")
    b = pd.Series([3.14, np.nan, 2.71828], dtype="Float32", name="float32")
    c = pd.Series([True, False, None], dtype="boolean", name="bool_nullable")
    d = pd.Series(["foo", None, "bar"], dtype="string", name="string")
    data = ["1678-01-01", "2262-04-11"]
    series = pd.Series(data, name="datetime_series")
    e = pd.to_datetime(series, utc=False)
    e.name = "datetime"
    data = ["1 day", None, "1 sec", "02:00:00"]
    series = pd.Series(data, name="timedelta_series")
    f = pd.to_timedelta(series)
    f.name = "timedelta"
    g = pd.Series([{"one": 1}, None, [{1, 2}]], dtype="object", name="object")
    h = pd.Series([1, None, 3, 4], dtype="UInt32", name="uint_nullable")
    i = pd.Series(np.arange(5), name="arange")
    assert i.dtype == "int64"
    j = pd.Series(np.linspace(0, 1, 11), name="linspace")
    assert j.dtype == "float64"

    series = [a, a, b, c, d, e, f, g, h, i, j]

    # Test individual series
    for s in series:
        roundtrip_series(s)

    # Test series combined into one DataFrame
    df = pd.concat(series, axis=1)

    roundtrip_df(df)

    # DataFrame with duplicate column names
    df = pd.concat([a, a, b], axis=1)

    roundtrip_df(df)

    # DataFrame with renamed columns
    df = pd.concat([b, d], axis=1)

    df = df.rename(columns={"string": "d", "float32": "b"})

    # Check names
    for column, series in df.items():
        assert series.name == column

    roundtrip_df(df)

def test_categories():
    for categories in [
        [1, 2, 3, 4],
        [1, 2, 3, 4, None],
        ["red", "green", "blue", 123, None],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, None],
    ]:
        values = [random.choice(categories) for _ in range(50)]
        s = pd.Series(values, name="Adelheid").astype("category")

        roundtrip_series(s)

    dtype = pd.CategoricalDtype(["n", "b", "a"], ordered=True)
    s = pd.Series(list("banana"), dtype=dtype, name="Berthold")

    roundtrip_series(s)

    index = pd.CategoricalIndex([1, 2, 3], categories=[1, 2, 3], name="Conrad")

    roundtrip_index(index)

    series = pd.Series([1, 2, 3], name="Dietlinde", index=index)

    roundtrip_series(series)

def test_numpy_dtypes():
    for dtype in numpy_dtypes:
        data = [0, 1, 0, 1, 0, 0, 0, 1, 1, 1]
        s = pd.Series(data, dtype=dtype, name=f"numpy_{dtype}")
        roundtrip_series(s)

def test_datetime():
    df = pd.DataFrame({
        "year": [2025, 2026],
        "month": [1, 2],
        "day": [1, 2]})

    series = pd.to_datetime(df)
    series.name = "Eberhard"

    roundtrip_series(series)

    t = pd.to_timedelta("1 days 01:02:03.00004")
    series = pd.Series([t], name="Frieda")

    roundtrip_series(series)

    # Test datetime with timezone. Timezone-aware objects
    # should always be checked with and without timezone
    # because internally, Pandas often stores timestamps with
    # NumPy, which is not timezone aware, and the conversion is
    # easy to get wrong.
    start = pd.to_datetime("1/1/2025").tz_localize("Europe/Berlin")
    end = pd.to_datetime("12/31/2025").tz_localize("Europe/Berlin")
    index = pd.date_range(start=start, end=end, name="Gerhardt")

    series = pd.Series(index, name="Gisela")

    data = [pd.Timestamp("1/1/1970").tz_localize("Europe/Berlin")]
    df = pd.DataFrame({"s": data})
    roundtrip_df(df)

    index = pd.DatetimeIndex(data, name="Gunther")

    series = pd.Series(data, index=index, name="Hedwig")

    roundtrip_series(series)

    roundtrip_df(pd.DataFrame({"s": series}, index=index))

    index = pd.CategoricalIndex(data, name="Heinrich")

    series = pd.Series(data, index=index, name="Irmgard")

    roundtrip_series(series)

    df = pd.DataFrame({"s": series}, index=index)

    roundtrip_df(df)

def test_datetime_index_naive():
    index = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    index.name = "Karl"
    roundtrip_index(index)

def test_timedelta_index():
    index = pd.timedelta_range(
        start="1 day",
        end="5 days",
        freq="D",
        name="Hugo")

    roundtrip_index(index)
    series = pd.Series([1, 2, 3, 4, 5], index=index)
    roundtrip_series(series)
    df = pd.DataFrame({"values": [1, 2, 3, 4, 5]}, index=index)
    roundtrip_df(df)

def test_interval_index():
    index = pd.interval_range(
        start=0,
        end=20,
        freq=3,
        closed="both",
        name="Mathilde")

    serialized_data = dumps(index)
    deserialized_index = loads(serialized_data)
    pd.testing.assert_index_equal(index, deserialized_index)

    data = [1, 2, 3, 4, 5, 6]
    series = pd.Series(data, index=index, name="Oswald")
    roundtrip_series(series)

    df = pd.DataFrame({"values": data}, index=index)
    roundtrip_df(df)

    for i in range(6):
        interval = df.index[i]
        assert df.loc[interval, "values"] == data[i]

    test_interval = pd.Interval(0, 3, closed="both")
    assert test_interval in df.index

def test_interval_index_with_datetime():
    start = pd.to_datetime("1/1/2025").tz_localize("Europe/Berlin")
    end = pd.to_datetime("12/31/2025").tz_localize("Europe/Berlin")
    index = pd.interval_range(
        start=start,
        end=end,
        freq="2D",
        closed="both",
        name="Reinhilde")

    roundtrip_index(index)

    data = [start + pd.Timedelta(days=i * 2) for i in range(len(index))]

    roundtrip_series(pd.Series(data, index=index, name="Siegfried"))

    df = pd.DataFrame({"dates": data}, index=index)

    roundtrip_df(df)

def test_period_index():
    index = pd.period_range(
        start="2000-01-01",
        end="2001-01-01",
        freq="M",
        name="Egon")

    roundtrip_index(index)

    series = pd.Series(range(len(index)), index=index, name="Hugo")
    roundtrip_series(series)

    df = pd.DataFrame({"values": range(len(index))}, index=index)
    roundtrip_df(df)

def test_multi_index():
    arrays = [[1, 1, 2, 2], ["red", "blue", "red", "blue"]]
    names = ["number", "color"]
    index = pd.MultiIndex.from_arrays(arrays, sortorder=1, names=names)
    roundtrip_index(index)

    series = pd.Series(np.random.randn(4), index=index, name="Manfred")
    roundtrip_series(series)

    df = pd.DataFrame({"values": np.random.randn(4)}, index=index)
    roundtrip_df(df)

def test_categorical_index_advanced():
    # With None in data
    categories = ["a", "b", "c"]
    data = ["a", "b", "a", None, "c"]
    index = pd.CategoricalIndex(data, categories=categories, name="with_none")
    roundtrip_index(index)

    # Ordered with custom order
    categories = ["low", "medium", "high"]
    data = ["medium", "high", "low", "medium"]
    index = pd.CategoricalIndex(
        data,
        categories=categories,
        ordered=True,
        name="ordered_custom")

    roundtrip_index(index)

    # Unused categories
    categories = ["apple", "banana", "cherry", "date"]
    data = ["apple", "cherry", "apple"]
    index = pd.CategoricalIndex(
        data,
        categories=categories,
        name="unused_categories")

    roundtrip_index(index)

    # Datetime with timezone
    tz = "America/New_York"
    dates = ["2023-01-01", "2023-01-02", "2023-01-03"]
    categories = pd.to_datetime(dates).tz_localize(tz)
    data = [categories[0], categories[2], categories[0]]
    index = pd.CategoricalIndex(
        data,
        categories=categories,
        name="datetime_tz")

    roundtrip_index(index)

    # Empty data
    index = pd.CategoricalIndex(
        [],
        categories=["x", "y", "z"],
        name="empty_with_categories")

    roundtrip_index(index)
