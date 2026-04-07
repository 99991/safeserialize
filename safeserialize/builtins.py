from . import *
import struct

TYPE_BOOL_FALSE = 0
TYPE_BOOL_TRUE = 1
TYPE_INT1 = 2
TYPE_INT2 = 3
TYPE_INT4 = 4
TYPE_INT8 = 5
TYPE_LARGE_INT = 6
TYPE_INT_VALUE_0 = 48 # 48 = ord('0')

@reader('builtins.int')
def read_int(ser, f):
    result = ser._read(f)

    if not isinstance(result, int):
        raise ValueError(f"Expected int, got {type(result)}")

    return result

def num_bytes_signed_int(data):
    bits = data.bit_length() or 1
    n = (bits + 7) // 8
    sign_bit = 1 << (8 * n - 1)

    # Need extra byte if sign bit is set
    if (data >= 0 and data >= sign_bit) or (data < 0 and data < -sign_bit):
        n += 1

    return n

@writer('builtins.int', raw = True)
def write_int(ser, data, out):
    # Ints are encoded with different type IDs based on their size.
    if 0 <= data < 10:
        out.write(bytes([TYPE_INT_VALUE_0 + data]))
    elif -128 <= data <= 127:
        out.write(bytes([TYPE_INT1]))
        out.write(struct.pack("<b", data))
    elif -32768 <= data <= 32767:
        out.write(bytes([TYPE_INT2]))
        out.write(struct.pack("<h", data))
    elif -2147483648 <= data <= 2147483647:
        out.write(bytes([TYPE_INT4]))
        out.write(struct.pack("<i", data))
    else:
        out.write(bytes([TYPE_LARGE_INT]))
        num_bytes = num_bytes_signed_int(data)
        out.write(struct.pack("<Q", num_bytes))
        out.write(data.to_bytes(num_bytes, byteorder="little", signed=True))

for i in range(10):
    def make_reader(i):
        f = lambda _ser, _: i
        f.__name__ = f'read_{i}'
        return f
    globals()[f'read_{i}'] = reader(None, TYPE_INT_VALUE_0 + i)(make_reader(i))
    
@reader(None, TYPE_INT1)
def read_int1(ser, f):
    return struct.unpack("<b", f.read(1))[0]

@reader(None, TYPE_INT2)
def read_int2(ser, f):
    return struct.unpack("<h", f.read(2))[0]

@reader(None, TYPE_INT4)
def read_int4(ser, f):
    return struct.unpack("<i", f.read(4))[0]

@reader(None, TYPE_LARGE_INT)
def read_large_int(ser, f):
    num_bytes, = struct.unpack("<Q", f.read(8))
    return int.from_bytes(f.read(num_bytes), byteorder="little", signed=True)

@writer("builtins.bool", raw = True)
def write_bool(ser, data, out):
    out.write(bytes([TYPE_BOOL_TRUE if data else TYPE_BOOL_FALSE]))

@reader(None, TYPE_BOOL_FALSE)
def read_false(ser, f):
    return False

@reader(None, TYPE_BOOL_TRUE)
def read_true(ser, f):
    return True

@writer("builtins.list", 23)
def write_list(ser, data, out):
    ser._write_int(len(data), out)
    for value in data:
        ser._write(value, out)

@reader("builtins.list", 23)
def read_list(ser, f):
    length = ser._read_int(f)
    return [ser._read(f) for _ in range(length)]

@writer("builtins.dict", 25)
def write_dict(ser, data, out):
    ser._write_int(len(data), out)
    for key, value in data.items():
        ser._write(key, out)
        ser._write(value, out)

@reader("builtins.dict", 25)
def read_dict(ser, f):
    length = ser._read_int(f)
    return {ser._read(f): ser._read(f) for _ in range(length)}

@writer("builtins.tuple", 24)
def write_tuple(ser, data, out):
    ser._write_int(len(data), out)
    for value in data:
        ser._write(value, out)

@reader("builtins.tuple", 24)
def read_tuple(ser, f):
    return tuple(read_list(ser, f))

@writer("builtins.set", 26)
def write_set(ser, data, out):
    ser._write_int(len(data), out)
    for value in data:
        ser._write(value, out)

@reader("builtins.set", 26)
def read_set(ser, f):
    return set(read_list(ser, f))

@writer("builtins.frozenset", 27)
def write_frozenset(ser, data, out):
    ser._write_int(len(data), out)
    for value in data:
        ser._write(value, out)

@reader("builtins.frozenset", 27)
def read_frozenset(ser, f):
    return frozenset(read_list(ser, f))

@writer("builtins.bytes", 21)
def write_bytes(ser, data, out):
    ser._write_int(len(data), out)
    out.write(data)

@reader("builtins.bytes", 21)
def read_bytes(ser, f):
    length = ser._read_int(f)
    return f.read(length)

@writer("builtins.bytearray", 22)
def write_bytearray(ser, data, out):
    ser._write_int(len(data), out)
    out.write(data)

@reader("builtins.bytearray", 22)
def read_bytearray(ser, f):
    length = ser._read_int(f)
    return bytearray(f.read(length))

@writer("builtins.str", 20)
def write_str(ser, data, out):
    write_bytes(ser, data.encode("utf-8"), out)

@reader("builtins.str", 20)
def read_str(ser, f):
    return read_bytes(ser, f).decode("utf-8")

@writer("builtins.float", 30)
def write_float(ser, data, out):
    out.write(struct.pack("<d", data))

@reader("builtins.float", 30)
def read_float(ser, f):
    return struct.unpack("<d", f.read(8))[0]

@writer("builtins.NoneType", 28)
def write_none(ser, data, out):
    pass

@reader("builtins.NoneType", 28)
def read_none(ser, f):
    return None

@writer("builtins.complex", 31)
def write_complex(ser, data, out):
    out.write(struct.pack("<dd", data.real, data.imag))

@reader("builtins.complex", 31)
def read_complex(ser, f):
    real, imag = struct.unpack("<dd", f.read(2 * 8))
    return complex(real, imag)

@writer("builtins.range", 34)
def write_range(ser, data, out):
    ser._write_int(data.start, out)
    ser._write_int(data.stop, out)
    ser._write_int(data.step, out)

@reader("builtins.range", 34)
def read_range(ser, f):
    start = ser._read_int(f)
    stop = ser._read_int(f)
    step = ser._read_int(f)
    return range(start, stop, step)

@writer("builtins.slice", 35)
def write_slice(ser, data, out):
    ser._write(data.start, out)
    ser._write(data.stop, out)
    ser._write(data.step, out)

@reader("builtins.slice", 35)
def read_slice(ser, f):
    start = ser._read(f)
    stop = ser._read(f)
    step = ser._read(f)
    return slice(start, stop, step)

@writer("builtins.ellipsis", 32)
def write_ellipsis(ser, data, out):
    pass

@reader("builtins.ellipsis", 32)
def read_ellipsis(ser, f):
    return Ellipsis

@writer("builtins.NotImplementedType", 33)
def write_not_implemented(ser, data, out):
    pass

@reader("builtins.NotImplementedType", 33)
def read_not_implemented(ser, f):
    return NotImplemented

@writer(BaseException, 36, sub = True)
def write_BaseException(ser, exc, out):
    ser._write_str(exc.__class__.__module__, out)
    ser._write_str(exc.__class__.__name__, out)
    ser._write_tuple(exc.args, out)
    ser._write_dict(exc.__dict__, out)

@reader('builtins.BaseException', 36)
def read_BaseException(ser, f):
    module_name = ser._read_str(f)
    cls_name = ser._read_str(f)
    args = ser._read_tuple(f)
    dct = ser._read_dict(f)
    import importlib
    module = importlib.import_module(module_name) 
    cls = getattr(module, cls_name)
    exc = cls()
    exc.args = args
    exc.__dict__ = dct
    return exc
