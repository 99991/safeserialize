from safeserialize import write, read
from ..core import writer, reader
from .numpy import _allowed_dtypes as _numpy_dtypes

VERSION = 1

_pandas_dtypes = {
    "boolean",
    "Int8", "Int16", "Int32", "Int64",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "Float32", "Float64",
}

@writer("pandas._libs.missing.NAType")
def write_na_type(data, out):
    write(VERSION, out)

@reader("pandas._libs.missing.NAType")
def read_na_type(f):
    version = read(f)
    assert version == VERSION
    import pandas as pd
    return pd.NA

@writer("pandas.core.indexes.range.RangeIndex")
def write_range_index(index, out):
    write(index.start, out)
    write(index.stop, out)
    write(index.step, out)

@reader("pandas.core.indexes.range.RangeIndex")
def read_range_index(f):
    start = read(f)
    stop = read(f)
    step = read(f)
    import pandas as pd
    return pd.RangeIndex(start, stop, step)

@writer("pandas.core.indexes.frozen.FrozenList")
def writer_frozen_list(data, out):
    write(list(data), out)

@reader("pandas.core.indexes.frozen.FrozenList")
def reader_frozen_list(f):
    data = read(f)
    import pandas
    return pandas.core.indexes.frozen.FrozenList(data)

@writer("pandas.core.indexes.base.Index")
def write_base_index(index, out):
    dtype = index.dtype
    dtype_name = dtype.name

    assert dtype_name in _numpy_dtypes
    write(index.name, out)
    write(dtype_name, out)
    write(index.names, out)
    write(index._data, out)

@reader("pandas.core.indexes.base.Index")
def reader_base_index(f):
    import pandas as pd

    name = read(f)
    dtype_name = read(f)
    assert dtype_name in _numpy_dtypes
    names = read(f)
    data = read(f)

    index = pd.Index(data, dtype=dtype_name, name=name)
    index.names = names
    return index

@writer("pandas.core.series.Series")
def write_series(series, out):
    import numpy
    import pandas

    values = series.values
    values_dtype_name = values.dtype.name

    write(VERSION, out)
    write(series.name, out)
    write(series.dtype, out)
    write(series.values.dtype, out)
    write(series.index, out)

    if values_dtype_name == "string":
        assert isinstance(values, pandas.core.arrays.string_.StringArray)
        write(values.tolist(), out)

    elif values_dtype_name in _pandas_dtypes:
        write(values.isna(), out)
        values_numpy = values._data
        assert isinstance(values_numpy, numpy.ndarray)
        write(values_numpy, out)

    elif values_dtype_name in _numpy_dtypes:
        assert isinstance(values, numpy.ndarray)
        write(values, out)

    elif values_dtype_name == "category":
        write(values.categories, out)
        assert isinstance(values.codes, numpy.ndarray)
        write(values.codes, out)
        write(values.ordered, out)

    else:
        raise ValueError(f"Pandas dtype {values_dtype_name} not implemented")

@reader("pandas.core.series.Series")
def read_series(f):
    import pandas as pd
    import numpy as np

    version = read(f)
    assert version == VERSION
    series_name = read(f)
    series_dtype = read(f)
    values_dtype = read(f)
    values_dtype_name = values_dtype.name
    index = read(f)

    if values_dtype_name == "string":
        values = read(f)
        array = pd.array(values, dtype="string")
        series = pd.Series(array, dtype="string", index=index)

    elif values_dtype_name in _numpy_dtypes:
        values = read(f)

        # NumPy datetime64[ns] does not have timezone information.
        # But pd.Series does, so if the series_dtype contains a timezone,
        # we have to make sure that we remove that and apply it later
        # or else pd.Series will change our times to account for the
        # timezone difference between NumPy (UTC by default) and pandas.
        if isinstance(series_dtype, pd.DatetimeTZDtype):
            # Create series from timezone-less dtype
            series = pd.Series(values, dtype=series_dtype.base, index=index)
            # and apply actual dtype afterwards
            series = series.dt.tz_localize("UTC").dt.tz_convert(series_dtype.tz)
        else:
            series = pd.Series(values, dtype=series_dtype, index=index)

    elif values_dtype_name in _pandas_dtypes:
        isna = read(f)
        assert isna.dtype == np.bool_
        values = read(f)
        series = pd.Series(values, dtype=series_dtype, index=index)
        series = series.mask(isna)

    elif values_dtype_name == "category":
        categories = read(f)
        codes = read(f)
        ordered = read(f) # unused, already stored in categories
        assert isinstance(ordered, bool)
        categorical = pd.Categorical.from_codes(codes, categories)
        series = pd.Series(categorical, dtype=series_dtype, index=index)

    else:
        raise ValueError(f"Pandas dtype {dtype_name} not implemented")

    series.name = series_name

    return series

@writer("pandas.core.frame.DataFrame")
def write_dataframe(data, out):
    write(VERSION, out)

    m, n = data.shape
    write(m, out)
    write(n, out)
    write(data.index, out)

    for _, series in data.items():
        write(series, out)

@reader("pandas.core.frame.DataFrame")
def read_dataframe(f):
    import pandas as pd

    version = read(f)
    assert version == VERSION

    m = read(f)
    n = read(f)
    index = read(f)

    series = [read(f) for _ in range(n)]

    df = pd.concat(series, axis=1)
    df.index = index

    assert df.shape == (m, n)

    return df

pandas_dtypes = [
    ("pandas.core.arrays.integer.Int8Dtype", "Int8Dtype"),
    ("pandas.core.arrays.integer.Int16Dtype", "Int16Dtype"),
    ("pandas.core.arrays.integer.Int32Dtype", "Int32Dtype"),
    ("pandas.core.arrays.integer.Int64Dtype", "Int64Dtype"),
    ("pandas.core.arrays.integer.UInt8Dtype", "UInt8Dtype"),
    ("pandas.core.arrays.integer.UInt16Dtype", "UInt16Dtype"),
    ("pandas.core.arrays.integer.UInt32Dtype", "UInt32Dtype"),
    ("pandas.core.arrays.integer.UInt64Dtype", "UInt64Dtype"),
    ("pandas.core.arrays.floating.Float32Dtype", "Float32Dtype"),
    ("pandas.core.arrays.floating.Float64Dtype", "Float64Dtype"),
    ("pandas.core.arrays.boolean.BooleanDtype", "BooleanDtype"),
    ("pandas.core.arrays.string_.StringDtype", "StringDtype"),
]

for dtype_path, dtype_name in pandas_dtypes:
    def make_dtype_reader_writer(dtype_path, dtype_name):
        @writer(dtype_path)
        def writer_func(data, out):
            pass

        @reader(dtype_path)
        def reader_func(f):
            import pandas as pd
            return getattr(pd, dtype_name)()

    make_dtype_reader_writer(dtype_path, dtype_name)

@writer("pandas.core.dtypes.dtypes.CategoricalDtype")
def write_CategoricalDtype(data, out):
    import pandas
    assert isinstance(data.categories, pandas.core.indexes.base.Index)
    write(data.categories, out)
    write(data.ordered, out)

@reader("pandas.core.dtypes.dtypes.CategoricalDtype")
def read_CategoricalDtype(f):
    import pandas
    categories = read(f)
    ordered = read(f)
    return pandas.CategoricalDtype(categories, ordered=ordered)

@writer("pandas.core.dtypes.dtypes.DatetimeTZDtype")
def write_DatetimeTZDtype(data, out):
    write(data.unit, out)
    write(data.tz, out)

@reader("pandas.core.dtypes.dtypes.DatetimeTZDtype")
def read_DatetimeTZDtype(f):
    unit = read(f)
    tz = read(f)
    import pandas
    return pandas.DatetimeTZDtype(unit, tz)

@writer("pandas._libs.tslibs.timedeltas.Timedelta")
def write_Timedelta(data, out):
    write(data.value, out)
    write(data.unit, out)

@reader("pandas._libs.tslibs.timedeltas.Timedelta")
def read_Timedelta(f):
    value = read(f)
    unit = read(f)
    import pandas
    return pandas.Timedelta(value, unit=unit)

@writer("pandas.core.indexes.datetimes.DatetimeIndex")
def write_DatetimeIndex(index, out):
    write(index.values, out)
    write(index.tz, out)
    write(index.name, out)

@reader("pandas.core.indexes.datetimes.DatetimeIndex")
def read_DatetimeIndex(f):
    import pandas as pd
    values = read(f)
    tz = read(f)
    name = read(f)
    values = pd.Series(values, name=name)
    values = values.dt.tz_localize("UTC").dt.tz_convert(tz)
    return pd.DatetimeIndex(values)

@writer("pandas.core.indexes.timedeltas.TimedeltaIndex")
def write_TimedeltaIndex(index, out):
    write(index.values, out)
    write(index.name, out)
    write(index.freqstr, out)

@reader("pandas.core.indexes.timedeltas.TimedeltaIndex")
def read_TimedeltaIndex(f):
    import pandas as pd
    values = read(f)
    name = read(f)
    freq = read(f)
    return pd.TimedeltaIndex(values, name=name, freq=freq)

# TODO implement other indexes listed here:
# https://pandas.pydata.org/docs/reference/api/pandas.Index.html

@writer("pandas.core.indexes.category.CategoricalIndex")
def write_CategoricalIndex(index, out):
    write(index.categories, out)
    write(index.ordered, out)
    write(index.name, out)

@reader("pandas.core.indexes.category.CategoricalIndex")
def read_CategoricalIndex(f):
    categories = read(f)
    ordered = read(f)
    name = read(f)
    import pandas as pd
    return pd.CategoricalIndex(categories, ordered=ordered, name=name)

@writer("pandas.core.indexes.interval.IntervalIndex")
def write_interval_index(index, out):
    write(index.left, out)
    write(index.right, out)
    write(index.closed, out)
    write(index.name, out)

@reader("pandas.core.indexes.interval.IntervalIndex")
def read_interval_index(f):
    left = read(f)
    right = read(f)
    closed = read(f)
    name = read(f)
    import pandas as pd
    return pd.IntervalIndex.from_arrays(left=left, right=right, closed=closed, name=name)

@writer("pandas.core.indexes.period.PeriodIndex")
def write_PeriodIndex(index, out):
    write(index.name, out)
    write(index.freqstr, out)
    write(index.asi8, out)

@reader("pandas.core.indexes.period.PeriodIndex")
def read_PeriodIndex(f):
    import pandas as pd
    name = read(f)
    freq = read(f)
    ordinals = read(f)
    return pd.PeriodIndex.from_ordinals(ordinals, freq=freq, name=name)
