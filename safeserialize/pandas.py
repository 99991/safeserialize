from . import *
from . import pytz, writer as core_writer, reader as core_reader
from .numpy import _allowed_dtypes as _numpy_dtypes
import warnings

VERSION = 1

_pandas_dtypes = {
    "boolean",
    "Int8", "Int16", "Int32", "Int64",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "Float32", "Float64",
}

_WARNING_SHOWN = False

def _warn_experimental():
    global _WARNING_SHOWN
    if _WARNING_SHOWN:
        return

    _WARNING_SHOWN = True

    warning_message = (
        "Serialization of Pandas objects is still experimental. "
        "The binary format can change at any time. "
        "Please verify that loads(dumps(data)) returns your original data "
        "and report any bugs you might encounter: "
        "https://github.com/99991/safeserialize/issues"
    )
    warnings.warn(warning_message, UserWarning, stacklevel=5)

def writer(type_str):
    original_writer = core_writer(type_str)
    def decorator(func):
        def wrapper(*args, **kwargs):
            _warn_experimental()
            return func(*args, **kwargs)
        wrapped = original_writer(wrapper)
        wrapped.__name__ = 'wrapped_' + func.__name__
        return wrapped
    return decorator

def reader(type_str):
    original_reader = core_reader(type_str)
    def decorator(func):
        def wrapper(*args, **kwargs):
            _warn_experimental()
            return func(*args, **kwargs)
        wrapped = original_reader(wrapper)
        wrapped.__name__ = 'wrapped_' + func.__name__
        return wrapped
    return decorator

@writer("pandas._libs.missing.NAType")
def write_na_type(ser, data, out):
    ser._write(VERSION, out)

@reader("pandas._libs.missing.NAType")
def read_na_type(ser, f):
    version = ser._read(f)
    assert version == VERSION
    import pandas as pd
    return pd.NA

@writer("pandas.core.indexes.range.RangeIndex")
def write_range_index(ser, index, out):
    ser._write(index.start, out)
    ser._write(index.stop, out)
    ser._write(index.step, out)

@reader("pandas.core.indexes.range.RangeIndex")
def read_range_index(ser, f):
    start = ser._read(f)
    stop = ser._read(f)
    step = ser._read(f)
    import pandas as pd
    return pd.RangeIndex(start, stop, step)

@writer("pandas.core.indexes.frozen.FrozenList")
def writer_frozen_list(ser, data, out):
    ser._write(list(data), out)

@reader("pandas.core.indexes.frozen.FrozenList")
def reader_frozen_list(ser, f):
    data = ser._read(f)
    import pandas
    return pandas.core.indexes.frozen.FrozenList(data)

@writer("pandas.core.indexes.base.Index")
def write_base_index(ser, index, out):
    dtype = index.dtype
    dtype_name = dtype.name

    assert dtype_name in _numpy_dtypes
    ser._write(index.name, out)
    ser._write(dtype_name, out)
    ser._write(index.names, out)
    ser._write(index._data, out)

@reader("pandas.core.indexes.base.Index")
def reader_base_index(ser, f):
    import pandas as pd

    name = ser._read(f)
    dtype_name = ser._read(f)
    assert dtype_name in _numpy_dtypes
    names = ser._read(f)
    data = ser._read(f)

    index = pd.Index(data, dtype=dtype_name, name=name)
    index.names = names
    return index

@writer("pandas.core.series.Series")
def write_series(ser, series, out):
    import numpy
    import pandas

    values = series.values
    values_dtype_name = values.dtype.name

    ser._write(VERSION, out)
    ser._write(series.name, out)
    ser._write(series.dtype, out)
    ser._write(series.values.dtype, out)
    ser._write(series.index, out)

    if values_dtype_name == "string":
        assert isinstance(values, pandas.core.arrays.string_.StringArray)
        ser._write(values.tolist(), out)

    elif values_dtype_name in _pandas_dtypes:
        ser._write(values.isna(), out)
        values_numpy = values._data
        assert isinstance(values_numpy, numpy.ndarray)
        ser._write(values_numpy, out)

    elif values_dtype_name in _numpy_dtypes:
        assert isinstance(values, numpy.ndarray)
        ser._write(values, out)

    elif values_dtype_name == "category":
        ser._write(values.categories, out)
        assert isinstance(values.codes, numpy.ndarray)
        ser._write(values.codes, out)
        ser._write(values.ordered, out)

    else:
        raise ValueError(f"Pandas dtype {values_dtype_name} not implemented")

@reader("pandas.core.series.Series")
def read_series(ser, f):
    import pandas as pd
    import numpy as np

    version = ser._read(f)
    assert version == VERSION
    series_name = ser._read(f)
    series_dtype = ser._read(f)
    values_dtype = ser._read(f)
    values_dtype_name = values_dtype.name
    index = ser._read(f)

    if values_dtype_name == "string":
        values = ser._read(f)
        array = pd.array(values, dtype="string")
        series = pd.Series(array, dtype="string", index=index)

    elif values_dtype_name in _numpy_dtypes:
        values = ser._read(f)

        # NumPy datetime64[ns] does not have timezone information.
        # But pd.Series does, so if the series_dtype contains a timezone,
        # we have to make sure that we remove that and apply it later
        # or else pd.Series will change our times to account for the
        # timezone difference between NumPy (UTC by default) and pandas.
        if isinstance(series_dtype, pd.DatetimeTZDtype):
            # Create series from timezone-less dtype
            series = pd.Series(values, dtype=series_dtype.base, index=index)
            # and apply actual dtype afterwards
            series = series.dt.tz_localize("UTC")
            series = series.dt.tz_convert(series_dtype.tz)
        else:
            series = pd.Series(values, dtype=series_dtype, index=index)

    elif values_dtype_name in _pandas_dtypes:
        isna = ser._read(f)
        assert isna.dtype == np.bool_
        values = ser._read(f)
        series = pd.Series(values, dtype=series_dtype, index=index)
        series = series.mask(isna)

    elif values_dtype_name == "category":
        categories = ser._read(f)
        codes = ser._read(f)
        # `ordered` is unused, already stored in categories
        ordered = ser._read(f)
        assert isinstance(ordered, bool)
        categorical = pd.Categorical.from_codes(codes, categories)
        series = pd.Series(categorical, dtype=series_dtype, index=index)

    else:
        raise ValueError(f"Pandas dtype {dtype_name} not implemented")

    series.name = series_name

    return series

@writer("pandas.core.frame.DataFrame")
def write_dataframe(ser, data, out):
    ser._write(VERSION, out)

    m, n = data.shape
    ser._write(m, out)
    ser._write(n, out)
    ser._write(data.index, out)

    for _, series in data.items():
        ser._write(series, out)

@reader("pandas.core.frame.DataFrame")
def read_dataframe(ser, f):
    import pandas as pd

    version = ser._read(f)
    assert version == VERSION

    m = ser._read(f)
    n = ser._read(f)
    index = ser._read(f)

    series = [ser._read(f) for _ in range(n)]

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

def make_dtype_reader_writer(dtype_path, dtype_name):
    @writer(dtype_path)
    def write_pandas_dtype(ser, data, out):
        pass

    @reader(dtype_path)
    def read_pandas_dtype(ser, f):
        import pandas as pd
        return getattr(pd, dtype_name)()

    read_pandas_dtype.__name__ += f'_{dtype_path}'
    globals().update({
        f"write_pandas_dtype_{dtype_path}": write_pandas_dtype,
        read_pandas_dtype.__name__: read_pandas_dtype,
    })    
        
for dtype_path, dtype_name in pandas_dtypes:
    make_dtype_reader_writer(dtype_path, dtype_name)

@writer("pandas.core.dtypes.dtypes.CategoricalDtype")
def write_CategoricalDtype(ser, data, out):
    import pandas
    assert isinstance(data.categories, pandas.core.indexes.base.Index)
    ser._write(data.categories, out)
    ser._write(data.ordered, out)

@reader("pandas.core.dtypes.dtypes.CategoricalDtype")
def read_CategoricalDtype(ser, f):
    import pandas
    categories = ser._read(f)
    ordered = ser._read(f)
    return pandas.CategoricalDtype(categories, ordered=ordered)

@writer("pandas.core.dtypes.dtypes.DatetimeTZDtype")
def write_DatetimeTZDtype(ser, data, out):
    ser._write(data.unit, out)
    ser._write(data.tz, out)

@reader("pandas.core.dtypes.dtypes.DatetimeTZDtype")
def read_DatetimeTZDtype(ser, f):
    unit = ser._read(f)
    tz = ser._read(f)
    import pandas
    return pandas.DatetimeTZDtype(unit, tz)

@writer("pandas._libs.tslibs.timedeltas.Timedelta")
def write_Timedelta(ser, data, out):
    ser._write(data.value, out)
    ser._write(data.unit, out)

@reader("pandas._libs.tslibs.timedeltas.Timedelta")
def read_Timedelta(ser, f):
    value = ser._read(f)
    unit = ser._read(f)
    import pandas
    return pandas.Timedelta(value, unit=unit)

@writer("pandas.core.indexes.datetimes.DatetimeIndex")
def write_DatetimeIndex(ser, index, out):
    ser._write(index.values, out)
    ser._write(index.tz, out)
    ser._write(index.name, out)

@reader("pandas.core.indexes.datetimes.DatetimeIndex")
def read_DatetimeIndex(ser, f):
    import pandas as pd
    values = ser._read(f)
    tz = ser._read(f)
    name = ser._read(f)
    values = pd.Series(values, name=name)
    values = values.dt.tz_localize("UTC").dt.tz_convert(tz)
    return pd.DatetimeIndex(values)

@writer("pandas.core.indexes.timedeltas.TimedeltaIndex")
def write_TimedeltaIndex(ser, index, out):
    ser._write(index.values, out)
    ser._write(index.name, out)
    ser._write(index.freqstr, out)

@reader("pandas.core.indexes.timedeltas.TimedeltaIndex")
def read_TimedeltaIndex(ser, f):
    import pandas as pd
    values = ser._read(f)
    name = ser._read(f)
    freq = ser._read(f)
    return pd.TimedeltaIndex(values, name=name, freq=freq)

@writer("pandas.core.indexes.category.CategoricalIndex")
def write_CategoricalIndex(ser, index, out):
    ser._write(index.codes, out)
    ser._write(index.categories, out)
    ser._write(index.ordered, out)
    ser._write(index.name, out)

@reader("pandas.core.indexes.category.CategoricalIndex")
def read_CategoricalIndex(ser, f):
    codes = ser._read(f)
    categories = ser._read(f)
    ordered = ser._read(f)
    name = ser._read(f)
    import pandas as pd
    cat = pd.Categorical.from_codes(codes, categories, ordered=ordered)
    return pd.CategoricalIndex(cat, name=name)

@writer("pandas.core.indexes.interval.IntervalIndex")
def write_interval_index(ser, index, out):
    ser._write(index.left, out)
    ser._write(index.right, out)
    ser._write(index.closed, out)
    ser._write(index.name, out)

@reader("pandas.core.indexes.interval.IntervalIndex")
def read_interval_index(ser, f):
    left = ser._read(f)
    right = ser._read(f)
    closed = ser._read(f)
    name = ser._read(f)
    import pandas as pd
    return pd.IntervalIndex.from_arrays(
        left=left,
        right=right,
        closed=closed,
        name=name)

@writer("pandas.core.indexes.multi.MultiIndex")
def write_MultiIndex(ser, index, out):
    ser._write(index.levels, out)
    ser._write(index.codes, out)
    ser._write(index.names, out)
    ser._write(index.sortorder, out)

@reader("pandas.core.indexes.multi.MultiIndex")
def read_MultiIndex(ser, f):
    import pandas as pd
    levels = ser._read(f)
    codes = ser._read(f)
    names = ser._read(f)
    sortorder = ser._read(f)
    return pd.MultiIndex(
        levels=levels,
        codes=codes,
        names=names,
        sortorder=sortorder)

@writer("pandas.core.indexes.period.PeriodIndex")
def write_PeriodIndex(ser, index, out):
    ser._write(index.name, out)
    ser._write(index.freqstr, out)
    ser._write(index.asi8, out)

@reader("pandas.core.indexes.period.PeriodIndex")
def read_PeriodIndex(ser, f):
    import pandas as pd
    name = ser._read(f)
    freq = ser._read(f)
    ordinals = ser._read(f)
    return pd.PeriodIndex.from_ordinals(ordinals, freq=freq, name=name)

serializer.harvest(globals())
#serializer.debug()
