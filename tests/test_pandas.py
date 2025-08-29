from safeserialize import dumps, loads
import pandas as pd
import numpy as np
import random
from safeserialize.types.numpy import _allowed_dtypes as numpy_dtypes

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
    a = pd.Series([1, 2, None, 4], dtype="Int64", name="int_nullable")
    b = pd.Series([3.14, np.nan, 2.71828], dtype="Float32", name="float32")
    c = pd.Series([True, False, None], dtype="boolean", name="bool_nullable")
    d = pd.Series(["foo", None, "bar"], dtype="string", name="string")
    e = pd.to_datetime(pd.Series(["1678-01-01", "2262-04-11"], name="datetime_internal"), utc=False)
    e.name = "datetime"
    f = pd.to_timedelta(pd.Series(["1 day", None, "1 minute", "02:00:00"], name="timedelta_internal"))
    f.name = "timedelta"
    g = pd.Series([{"one": 1}, None, [{1, 2}]], dtype="object", name="object")
    h = pd.Series([1, None, 3, 4], dtype="UInt32", name="uint_nullable")
    i = pd.Series(np.arange(5), name="arange")
    assert i.dtype == "int64"
    j = pd.Series(np.linspace(0, 1, 11), name="linspace")
    assert j.dtype == "float64"

    series = [a, a, b, c, d, e, f, g, h, i, j]

    for s in series:
        roundtrip_series(s)

    df = pd.concat(series, axis=1)

    roundtrip_df(df)

    # Data frame with duplicate column names
    df = pd.concat([a, a, b], axis=1)

    roundtrip_df(df)

    # Data frame with renamed columns
    df = pd.concat([b, d], axis=1)

    df = df.rename(columns={"string": "d", "float32": "b"})

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

    series = pd.Series([1, 2, 3], name="Dietlinde", index=index)

    roundtrip_series(series)

def test_numpy_dtypes():
    for dtype in numpy_dtypes:
        s = pd.Series([0, 1, 0, 1, 0, 0, 0, 1, 1, 1], dtype=dtype, name=f"numpy_{dtype}")
        roundtrip_series(s)


def test_datetime():
    df = pd.DataFrame({
        "year": [2025, 2026],
        "month": [1, 2],
        "day": [1, 2]})

    series = pd.to_datetime(df)
    series.name = "Eberhard"

    roundtrip_series(series)

    series = pd.Series([pd.to_timedelta('1 days 01:02:03.00004')], name="Frieda")

    roundtrip_series(series)

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

def test_timedelta_index():
    index = pd.timedelta_range(start="1 day", end="5 days", freq="D", name="Hugo")
    roundtrip_index(index)
    series = pd.Series([1, 2, 3, 4, 5], index=index)
    roundtrip_series(series)
    df = pd.DataFrame({"values": [1, 2, 3, 4, 5]}, index=index)
    roundtrip_df(df)

def test_interval_index():
    index = pd.interval_range(start=0, end=20, freq=3, closed="both", name="Mathilde")
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
    index = pd.interval_range(start=start, end=end, freq="2D", closed="both", name="Reinhilde")

    roundtrip_index(index)

    data = [start + pd.Timedelta(days=i*2) for i in range(len(index))]

    roundtrip_series(pd.Series(data, index=index, name="Siegfried"))

    df = pd.DataFrame({"dates": data}, index=index)

    roundtrip_df(df)