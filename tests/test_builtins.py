from .roundtrip import *
import tempfile

def test_num_bytes_signed_int():
    from safeserialize.builtins import num_bytes_signed_int
    assert num_bytes_signed_int(1) == 1
    assert num_bytes_signed_int(-1) == 1
    assert num_bytes_signed_int(127) == 1
    assert num_bytes_signed_int(-128) == 1
    assert num_bytes_signed_int(128) == 2
    assert num_bytes_signed_int(-129) == 2
    assert num_bytes_signed_int(32767) == 2
    assert num_bytes_signed_int(32768) == 3
    assert num_bytes_signed_int(-32768) == 2
    assert num_bytes_signed_int(-32769) == 3
    assert num_bytes_signed_int(2147483647) == 4
    assert num_bytes_signed_int(-2147483648) == 4

def test_builtins():
    
    from safeserialize import dump, load, dumps, loads
    

    for x in range(-300, 300):
        roundtrip(x)

    data = {
        1: [1, 2.0, 3, float("inf"), float("-inf")],
        3: [4, 5, 6],
        (1, 2): 3,
        # max one member because of test_binary_compat: frozenset does not guarantee order
        frozenset([7]): 4,
        123456789: 1 << 256,
        b"key": bytearray(b"value"),
        "float": 3.14159265358979323846,
        "true": True,
        "None": None,
        "list": [[], [[[]], []], [], [[], [[[]]]]],
        "set": {True, False, None, 1, 2, 3},
        "complex": 1 + 2j,
        "range": range(2, 10, -1),
        "slice": slice(None, 0.5, 123),
        "ellipsis": Ellipsis,
        "NotImplemented": NotImplemented,
    }

    roundtrip(data)

    # Test multi-member frozenset separately, avoiding the roundtrip function
    data = frozenset([7, "foo", 9])
    deserialized = loads(dumps(data))
    assert data == deserialized, f"{data} != {deserialized}"

    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        dump(data, temp_file)
        temp_file.seek(0)
        deserialized = load(temp_file)

    assert data == deserialized, f"{data} != {deserialized}"

def test_headerless():

    from safeserialize import dumps, loads
    
    data = 1
    assert len(roundtrip(data, header=False)) == 1

def test_constants():
    roundtrip_const(True)
    roundtrip_const(False)
    roundtrip_const(None)
    roundtrip_const(...)
    roundtrip_const(NotImplemented)

def test_exceptions():
    roundtrip_exc(RuntimeError("test error"))
