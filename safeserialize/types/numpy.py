import struct
import ast

from ..core import writer, reader, write_list, read_list

_allowed_dtypes = {
    "bool",
    "uint8", "uint16", "uint32", "uint64",
    "int8", "int16", "int32", "int64",
    "float16", "float32", "float64",
    "complex64", "complex128",
    "datetime64[ns]", "timedelta64[ns]",
    "object",
}

@writer("numpy.ndarray")
def write_ndarray(data, out, alignment=64):
    dtype = str(data.dtype)

    if dtype not in _allowed_dtypes:
        raise NotImplementedError(f"NumPy dtype {data.dtype} not supported")

    if alignment is not None:
        # Pad with leading zeros until aligned.
        # This is useful for some assembly instructions,
        # which require that kind of alignment.
        # Also, NumPy tries to align data to 64 byte boundary,
        # so we try to preserve that.
        pos = out.tell()

        if pos % alignment != 0:
            out.write(b"\x00" * (alignment - pos % alignment))

    magic = b"\x93NUMPY"

    version = bytes([1, 0])

    (_, descr), = data.dtype.descr

    header = "{"
    header += f"'descr': '{descr}', "
    header += "'fortran_order': False, "
    header += f"'shape': {data.shape}, "
    header += "}"
    header = header.encode("ascii")

    # magic + version + length + header (+ padding) + newline
    length = len(magic) + 2 + 2 + len(header) + 1

    padding = 0
    if (length + padding) % 64 != 0:
        padding = 64 - (length + padding) % 64

    header = b"".join([
        magic,
        version,
        struct.pack("<H", len(header) + padding + 1),
        header,
        b" " * padding,
        b"\n",
    ])

    assert len(header) % 64 == 0

    out.write(header)

    if dtype == "object":
        write_list(data.ravel(order="C"), out)
    else:
        out.write(data.tobytes())

@reader("numpy.ndarray")
def read_ndarray(f):
    import numpy as np

    magic = b"\x93NUMPY\1\0"

    buf = b""
    # Consume leading padding zeros
    for _ in range(10):
        buf += f.read(len(magic))

        idx = buf.find(magic[:1])

        # If we got the first byte of magic,
        # we can compute how much more magic we need to read.
        if idx != -1:
            # How much magic do we have already?
            have = len(buf) - idx
            remaining = len(magic) - have
            buf += f.read(remaining)
            break

    assert buf.lstrip(b"\0") == magic

    header_len, = struct.unpack("<H", f.read(2))

    assert (10 + header_len) % 64 == 0

    dict_data = f.read(header_len)

    assert dict_data.endswith(b"\n")

    d = ast.literal_eval(dict_data.decode("utf-8"))

    assert list(d) == ["descr", "fortran_order", "shape"]
    assert d["fortran_order"] is False

    descr = d["descr"]
    shape = d["shape"]

    assert descr in {
        "|b1",
        "|i1", "<i2", "<i4", "<i8",
        "|u1", "<u2", "<u4", "<u8",
        "<f2", "<f4", "<f8",
        "<c8", "<c16",
        "<m8[ns]",
        "<M8[ns]",
        "|O",
    }, f"NumPy dtype {repr(descr)} not implemented"

    if descr == "|O":
        objects = read_list(f)
        return np.array(objects, dtype=object).reshape(shape)
    else:
        itemsize = int(descr[2:4].rstrip("["))

        num_bytes = np.prod(shape) * itemsize

        buf = f.read(num_bytes)

        return np.frombuffer(buf, dtype=descr).reshape(shape).copy()

@writer("numpy.bool")
def write_bool(data, out):
    out.write(struct.pack("<?", data))

@reader("numpy.bool")
def read_bool(f):
    import numpy as np
    return np.bool_(struct.unpack("<?", f.read(1))[0])

@writer("numpy.int32")
def write_int32(data, out):
    out.write(struct.pack("<i", data))

@reader("numpy.int32")
def read_int32(f):
    import numpy as np
    return np.int32(struct.unpack("<i", f.read(4))[0])

@writer("numpy.int64")
def write_int64(data, out):
    out.write(struct.pack("<q", data))

@reader("numpy.int64")
def read_int64(f):
    import numpy as np
    return np.int64(struct.unpack("<q", f.read(8))[0])

@writer("numpy.float16")
def write_float16(data, out):
    out.write(struct.pack("<e", data))

@reader("numpy.float16")
def read_float16(f):
    import numpy as np
    return np.float16(struct.unpack("<e", f.read(2))[0])

@writer("numpy.float32")
def write_float32(data, out):
    out.write(struct.pack("<f", data))

@reader("numpy.float32")
def read_float32(f):
    import numpy as np
    return np.float32(struct.unpack("<f", f.read(4))[0])

@writer("numpy.float64")
def write_float64(data, out):
    out.write(struct.pack("<d", data))

@reader("numpy.float64")
def read_float64(f):
    import numpy as np
    return np.float64(struct.unpack("<d", f.read(8))[0])

@writer("numpy.complex64")
def write_complex64(data, out):
    out.write(struct.pack("<ff", data.real, data.imag))

@reader("numpy.complex64")
def read_complex64(f):
    import numpy as np
    real, imag = struct.unpack("<ff", f.read(8))
    return np.complex64(real + 1j * imag)

@writer("numpy.complex128")
def write_complex128(data, out):
    out.write(struct.pack("<dd", data.real, data.imag))

@reader("numpy.complex128")
def read_complex128(f):
    import numpy as np
    real, imag = struct.unpack("<dd", f.read(16))
    return np.complex128(real + 1j * imag)

@writer("numpy.int8")
def write_int8(data, out):
    out.write(struct.pack("<b", data))

@reader("numpy.int8")
def read_int8(f):
    import numpy as np
    return np.int8(struct.unpack("<b", f.read(1))[0])

@writer("numpy.int16")
def write_int16(data, out):
    out.write(struct.pack("<h", data))

@reader("numpy.int16")
def read_int16(f):
    import numpy as np
    return np.int16(struct.unpack("<h", f.read(2))[0])

@writer("numpy.uint8")
def write_uint8(data, out):
    out.write(struct.pack("<B", data))

@reader("numpy.uint8")
def read_uint8(f):
    import numpy as np
    return np.uint8(struct.unpack("<B", f.read(1))[0])

@writer("numpy.uint16")
def write_uint16(data, out):
    out.write(struct.pack("<H", data))

@reader("numpy.uint16")
def read_uint16(f):
    import numpy as np
    return np.uint16(struct.unpack("<H", f.read(2))[0])

@writer("numpy.uint32")
def write_uint32(data, out):
    out.write(struct.pack("<I", data))

@reader("numpy.uint32")
def read_uint32(f):
    import numpy as np
    return np.uint32(struct.unpack("<I", f.read(4))[0])

@writer("numpy.uint64")
def write_uint64(data, out):
    out.write(struct.pack("<Q", data))

@reader("numpy.uint64")
def read_uint64(f):
    import numpy as np
    return np.uint64(struct.unpack("<Q", f.read(8))[0])
