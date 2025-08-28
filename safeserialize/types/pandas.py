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
    write(dtype.name, out)
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
    dtype = values.dtype
    dtype_name = dtype.name

    write(VERSION, out)
    write(series.name, out)
    write(dtype_name, out)

    if dtype_name == "string":
        assert isinstance(values, pandas.core.arrays.string_.StringArray)
        write(values.tolist(), out)

    elif dtype_name in _pandas_dtypes:
        write(values.isna(), out)
        values_numpy = values._data
        assert isinstance(values_numpy, numpy.ndarray)
        write(values_numpy, out)

    elif dtype_name in _numpy_dtypes:
        assert isinstance(values, numpy.ndarray)
        write(values, out)

    elif dtype_name == "category":
        write(values.categories, out)
        assert isinstance(values.codes, numpy.ndarray)
        write(values.codes, out)
        write(values.ordered, out)

    else:
        raise ValueError(f"Pandas dtype {dtype_name} not implemented")

@reader("pandas.core.series.Series")
def read_series(f):
    import pandas as pd
    import numpy as np

    version = read(f)
    assert version == VERSION
    series_name = read(f)
    dtype_name = read(f)

    if dtype_name == "string":
        values = read(f)
        array = pd.array(values, dtype="string")
        series = pd.Series(array, dtype="string")

    elif dtype_name in _numpy_dtypes:
        values_np = read(f)
        series = pd.Series(values_np, dtype=dtype_name)

    elif dtype_name in _pandas_dtypes:
        isna = read(f)
        assert isna.dtype == np.bool_
        values_np = read(f)
        series = pd.Series(values_np, dtype=dtype_name)
        series = series.mask(isna)

    elif dtype_name == "category":
        categories = read(f)
        codes = read(f)
        ordered = read(f)
        categorical = pd.Categorical.from_codes(
            codes=codes,
            categories=categories,
            ordered=ordered,
            dtype=dtype_name)
        series = pd.Series(categorical)

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
