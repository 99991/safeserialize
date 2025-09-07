import io
import base64
import struct

from .constants import (
    builtins,
    VERSION,
    FILE_SIGNATURE,
    TYPE_BOOL_FALSE,
    TYPE_BOOL_TRUE,
    TYPE_INT1,
    TYPE_INT2,
    TYPE_INT4,
    TYPE_LARGE_INT,
    TYPE_CUSTOM,
    TYPE_INT_VALUE_0,
)

writers = {}
readers = {}

def writer(type_name):
    """Register a writer function for a type-name."""
    def decorator(func):
        assert type_name not in writers
        writers[type_name] = func
        return func
    return decorator

def reader(type_name):
    """Register a reader function for a type-name."""
    def decorator(func):
        assert type_name not in readers
        readers[type_name] = func
        if type_name in builtins:
            readers[builtins[type_name]] = func
        return func
    return decorator

class Serializer:
    def __init__(self, file, writers=None, header=True, version=VERSION):
        if writers is None:
            writers = {}

        if header:
            file.write(FILE_SIGNATURE)
            file.write(struct.pack("<I", version))

        self._file = file
        self._version = version
        self._writers = writers

    def write(self, data: bytes):
        """This write function mimics file.write. It does not serialize."""
        self._file.write(data)

    def tell(self):
        return self._file.tell()

class Deserializer:
    def __init__(self, file, readers=None, header=True, version=None):
        if readers is None:
            readers = {}

        self._file = file
        self._readers = readers

        if header:
            signature = self.read(len(FILE_SIGNATURE))

            if signature != FILE_SIGNATURE:
                raise ValueError(f"Invalid file signature {repr(signature)}")

            version, = struct.unpack("<I", self.read(4))
        else:
            if version is None:
                version = VERSION

        self._version = version

    def read(self, n):
        """Read bytes from underlying file. Does not deserialize.
        Throws ValueError if too few bytes were read."""
        data = self._file.read(n)

        if len(data) != n:
            raise ValueError(f"Expected {n} bytes, got {len(data)}")

        return data

def write(data, out):
    """Serialize data object and write it to output stream."""

    if data is False:
        out.write(bytes([TYPE_BOOL_FALSE]))
        return

    if data is True:
        out.write(bytes([TYPE_BOOL_TRUE]))
        return

    # Ints always include their type byte,
    # so they have to be handled differently from the other types.
    if isinstance(data, int):
        write_int(data, out)
        return

    cls = data.__class__
    data_type = f"{cls.__module__}.{cls.__name__}"

    out_writers = getattr(out, "_writers", {})

    if data_type not in out_writers:
        if data_type not in writers:
            raise TypeError(f"Writer not implemented for {type(data)}")
        else:
            writer = writers[data_type]
    else:
        writer = out_writers[data_type]

    type_id = builtins.get(data_type, TYPE_CUSTOM)

    out.write(bytes([type_id]))

    # Custom types get a type name
    if type_id == TYPE_CUSTOM:
        write_str(data_type, out)

    writer(data, out)

def read(f):
    """Read bytes from input stream and deserialize."""

    data_type, = f.read(1)

    f_readers = getattr(f, "_readers", {})

    if data_type in f_readers:
        return f_readers[data_type](f)

    elif data_type in readers:
        return readers[data_type](f)

    elif data_type == TYPE_CUSTOM:
        name = read_str(f)

        if name not in f_readers:
            if name not in readers:
                raise TypeError(f"Reader not implemented for {repr(name)}")
            else:
                reader = readers[name]
        else:
            reader = f_readers[name]

        return reader(f)
    else:
        raise TypeError(f"Reader not implemented for {repr(data_type)}")

def dump(obj, file, writers=None, header=True):
    """Serialize object to a file."""
    serializer = Serializer(file, writers, header)
    write(obj, serializer)

def dumps(obj, writers=None, header=True):
    """Serialize object to bytes."""
    out = io.BytesIO()
    serializer = Serializer(out, writers, header)
    write(obj, serializer)
    return out.getvalue()

def load(file, readers=None, header=True, version=None):
    """Deserialize object from a file."""
    deserializer = Deserializer(file, readers, header, version)
    return read(deserializer)

def loads(data, readers=None, header=True, version=None):
    """Deserialize object from a file."""
    deserializer = Deserializer(io.BytesIO(data), readers, header, version)
    return read(deserializer)

def dump_base64(data: bytes) -> str:
    """Like dumps, but encodes result as base64."""
    return base64.b64encode(dumps(data)).decode("ascii")

def load_base64(text: str) -> bytes:
    """Like loads, but decodes input from base64 beforehand."""
    return loads(base64.b64decode(text.encode("ascii")))

def read_int(f):
    result = read(f)

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

def write_int(data, out):
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
        return lambda _: i
    reader(TYPE_INT_VALUE_0 + i)(make_reader(i))

@reader(TYPE_INT1)
def read_int1(f):
    return struct.unpack("<b", f.read(1))[0]

@reader(TYPE_INT2)
def read_int2(f):
    return struct.unpack("<h", f.read(2))[0]

@reader(TYPE_INT4)
def read_int4(f):
    return struct.unpack("<i", f.read(4))[0]

@reader(TYPE_LARGE_INT)
def read_large_int(f):
    num_bytes, = struct.unpack("<Q", f.read(8))
    return int.from_bytes(f.read(num_bytes), byteorder="little", signed=True)

@reader(TYPE_BOOL_FALSE)
def read_false(f):
    return False

@reader(TYPE_BOOL_TRUE)
def read_true(f):
    return True

@writer("builtins.list")
def write_list(data, out):
    write_int(len(data), out)
    for value in data:
        write(value, out)

@reader("builtins.list")
def read_list(f):
    length = read_int(f)
    return [read(f) for _ in range(length)]

@writer("builtins.dict")
def write_dict(data, out):
    write_int(len(data), out)
    for key, value in data.items():
        write(key, out)
        write(value, out)

@reader("builtins.dict")
def read_dict(f):
    length = read_int(f)
    return {read(f): read(f) for _ in range(length)}

@writer("builtins.tuple")
def write_tuple(data, out):
    write_int(len(data), out)
    for value in data:
        write(value, out)

@reader("builtins.tuple")
def read_tuple(f):
    return tuple(read_list(f))

@writer("builtins.set")
def write_set(data, out):
    write_int(len(data), out)
    for value in data:
        write(value, out)

@reader("builtins.set")
def read_set(f):
    return set(read_list(f))

@writer("builtins.frozenset")
def write_frozenset(data, out):
    write_int(len(data), out)
    for value in data:
        write(value, out)

@reader("builtins.frozenset")
def read_frozenset(f):
    return frozenset(read_list(f))

@writer("builtins.bytes")
def write_bytes(data, out):
    write_int(len(data), out)
    out.write(data)

@reader("builtins.bytes")
def read_bytes(f):
    length = read_int(f)
    return f.read(length)

@writer("builtins.bytearray")
def write_bytearray(data, out):
    write_int(len(data), out)
    out.write(data)

@reader("builtins.bytearray")
def read_bytearray(f):
    length = read_int(f)
    return bytearray(f.read(length))

@writer("builtins.str")
def write_str(data, out):
    write_bytes(data.encode("utf-8"), out)

@reader("builtins.str")
def read_str(f):
    return read_bytes(f).decode("utf-8")

@writer("builtins.float")
def write_float(data, out):
    out.write(struct.pack("<d", data))

@reader("builtins.float")
def read_float(f):
    return struct.unpack("<d", f.read(8))[0]

@writer("builtins.bool")
def write_bool(data, out):
    out.write(bytes([1 if data else 0]))

@reader("builtins.bool")
def read_bool(f):
    return bool(f.read(1)[0])

@writer("builtins.NoneType")
def write_none(data, out):
    pass

@reader("builtins.NoneType")
def read_none(f):
    return None

@writer("builtins.complex")
def write_complex(data, out):
    out.write(struct.pack("<dd", data.real, data.imag))

@reader("builtins.complex")
def read_complex(f):
    real, imag = struct.unpack("<dd", f.read(2 * 8))
    return complex(real, imag)

@writer("builtins.range")
def write_range(data, out):
    write_int(data.start, out)
    write_int(data.stop, out)
    write_int(data.step, out)

@reader("builtins.range")
def read_range(f):
    start = read_int(f)
    stop = read_int(f)
    step = read_int(f)
    return range(start, stop, step)

@writer("builtins.slice")
def write_slice(data, out):
    write(data.start, out)
    write(data.stop, out)
    write(data.step, out)

@reader("builtins.slice")
def read_slice(f):
    start = read(f)
    stop = read(f)
    step = read(f)
    return slice(start, stop, step)

@writer("builtins.ellipsis")
def write_ellipsis(data, out):
    pass

@reader("builtins.ellipsis")
def read_ellipsis(f):
    return Ellipsis

@writer("builtins.NotImplementedType")
def write_not_implemented(data, out):
    pass

@reader("builtins.NotImplementedType")
def read_not_implemented(f):
    return NotImplemented
