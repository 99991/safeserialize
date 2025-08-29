VERSION = 1
# Adapted from PNG file signature
# https://www.libpng.org/pub/png/spec/1.2/PNG-Rationale.html#R.PNG-file-signature
FILE_SIGNATURE = b'\x89SER\r\n\x1a\n'

TYPE_BOOL_FALSE = 0
TYPE_BOOL_TRUE = 1
TYPE_INT1 = 2
TYPE_INT2 = 3
TYPE_INT4 = 4
TYPE_INT8 = 5
TYPE_LARGE_INT = 6
# 48 = ord('0')
TYPE_INT_VALUE_0 = 48

builtins = {
    # String/bytes
    "builtins.str": 20,
    "builtins.bytes": 21,
    "builtins.bytearray": 22,

    # Collections
    "builtins.list": 23,
    "builtins.tuple": 24,
    "builtins.dict": 25,
    "builtins.set": 26,
    "builtins.frozenset": 27,

    # Scalar
    "builtins.NoneType": 28,
    "builtins.bool": 29,
    "builtins.float": 30,
    "builtins.complex": 31,
    "builtins.ellipsis": 32,
    "builtins.NotImplementedType": 33,

    # Misc
    "builtins.range": 34,
    "builtins.slice": 35,
}

TYPE_CUSTOM = 255
