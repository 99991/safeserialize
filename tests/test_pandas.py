from safeserialize import dumps, loads
import pandas as pd
import numpy as np

def test_pandas():
    a = pd.Series([1, 2, None, 4], dtype="Int64", name="int_nullable")
    b = pd.Series([3.14, np.nan, 2.71828], dtype="Float32", name="float32")
    c = pd.Series([True, False, None], dtype="boolean", name="bool_nullable")
    d = pd.Series(["foo", None, "bar"], dtype="string", name="string")
    e = pd.to_datetime(pd.Series(["1678-01-01", "2262-04-11"]), utc=False)
    e.name = "datetime"
    f = pd.to_timedelta(pd.Series(["1 day", None, "1 minute", "02:00:00"]))
    f.name = "timedelta"
    g = pd.Series([{"one": 1}, None, [{1, 2}]], dtype="object", name="object")
    h = pd.Series([1, None, 3, 4], dtype="UInt32", name="uint_nullable")
    i = pd.Series(np.arange(5))
    assert i.dtype == "int64"
    j = pd.Series(np.linspace(0, 1, 11))
    assert j.dtype == "float64"

    series = [a, a, b, c, d, e, f, g, h, i, j]

    for s in series:
        serialized_data = dumps(s)

        deserialized_series = loads(serialized_data)

        pd.testing.assert_series_equal(s, deserialized_series)

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

def roundtrip_df(df):
    serialized_data = dumps(df)
    deserialized_df = loads(serialized_data)
    pd.testing.assert_frame_equal(df, deserialized_df)
