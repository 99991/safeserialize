from safeserialize import dumps, loads
import os, collections

DATA = collections.defaultdict(lambda: [])

def append(typ, obj):
    DATA[os.environ.get('PYTEST_CURRENT_TEST')].append( (typ,obj) )

def cmp_eq(x,y):
    assert x == y, f"{x} != {y}"

def cmp_is(x,y):
    assert x is y, f"{x} is not {y}"

def cmp_exc(x,y):
    try:
        raise y
    except type(x):
        assert x.args == y.args, f"{x} != {y}"
        assert x.__dict__ == y.__dict__, f"{x} != {y}"
        return
    except:
        pass
    raise AssertionError(f"{repr(x)} != {repr(y)}")

def cmp_array(x,y):
    import numpy as np
    return np.array_equal(x,y)

def cmp_sparse(x,y):
    assert (x != y).nnz == 0, f"{x} != {y}"

def cmp_series(x,y):
    import pandas as pd
    pd.testing.assert_series_equal(x,y)

def cmp_df(x,y):
    import pandas as pd
    pd.testing.assert_frame_equal(x,y)

def cmp_index(x,y):
    import pandas as pd
    pd.testing.assert_index_equal(x,y)

def cmp_torch(x,y):
    import torch
    assert torch.equal(x,y)

CMP_TYPES = {
    name[4:]: f
    for name, f in globals().items()
    if name.startswith('cmp_')
}

def _roundtrip(cmp_type, obj, header = True):
    append(cmp_type, obj)
    ser = dumps(obj, header = header)
    deser = loads(ser, header = header)
    assert_f = CMP_TYPES[cmp_type]
    assert_f(obj, deser)
    return ser
    
def roundtrip(obj, header=True):
    return _roundtrip("eq", obj, header)

def roundtrip_const(obj, header=True):
    return _roundtrip("is", obj, header)

def roundtrip_exc(obj, header=True):
    return _roundtrip("exc", obj, header)

def roundtrip_array(obj, header=True):
    return _roundtrip("array", obj, header)

def roundtrip_sparse(obj, header=True):
    return _roundtrip("sparse", obj, header)

def roundtrip_series(s, header=True):
    return _roundtrip("series", s, header)

def roundtrip_df(df, header=True):
    return _roundtrip("df", df, header)

def roundtrip_index(index, header=True):
    return _roundtrip("index", index, header)

def roundtrip_torch(x, header=True):
    return _roundtrip("torch", x, header)
