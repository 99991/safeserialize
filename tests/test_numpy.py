from safeserialize import dumps, loads
import io
import tempfile
import numpy as np
from safeserialize.types.numpy import write_ndarray, read_ndarray

def test_numpy_serialization():
    np.random.seed(0)

    x = np.array([1, 2, 3])

    data = {
        "3d array of doubles": np.random.rand(1, 2, 3),
        "1d array of ints": np.arange(5),
        "bool array": np.random.rand(5) < 0.5,
        "object array": np.array([{"one": 1}, None, {2: "two"}, {3, 4}]),
        "x.sum()": x.sum(),
        "bool": np.bool_(True),
        "int8": np.int8(-128),
        "int16": np.int16(-32768),
        "int32": np.int32(-2147483648),
        "int64": np.int64(-9223372036854775808),
        "uint8": np.uint8(255),
        "uint16": np.uint16(65535),
        "uint32": np.uint32(4294967295),
        "uint64": np.uint64(18446744073709551615),
        "float16": np.float16(2.71828),
        "float32": np.float32(3.14),
        "float64": np.float64(3.141592653589793),
        "complex64": np.complex64(1 + 2j),
        "complex128": np.complex128(1 + 2j),
    }

    serialized = dumps(data)

    deserialized = loads(serialized)

    for key, expected_value in data.items():
        value = deserialized[key]
        assert np.array_equal(value, expected_value)

def test_write_ndarray():
    # Verify write_ndarray against NumPy np.save
    np.random.seed(0)

    a = np.random.rand(3, 4, 5)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".npy") as tmp:
        np.save(tmp.name, a)
        tmp.seek(0)
        expected = tmp.read()

    f = io.BytesIO()
    write_ndarray(a, f)
    f.seek(0)
    data = f.read()

    assert data == expected

def test_read_ndarray():
    np.random.seed(0)

    a = np.random.rand(12, 13)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".npy") as tmp:
        np.save(tmp.name, a)
        tmp.seek(0)

        b = read_ndarray(tmp)

    assert np.array_equal(a, b)
    assert np.array_equal(a, b)
