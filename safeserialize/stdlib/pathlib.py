from .. import *

"""
On Linux, "a\\b" is a valid file name,
not a directory "a" with a file "b".
We convert to "a/b" with as_posix().
"""

@writer("pathlib._local.PosixPath")
def write_local_posix_path(ser, data, out):
    ser._write_str(data.as_posix(), out)

@reader("pathlib._local.PosixPath")
def read_local_posix_path(ser, f):
    from pathlib import Path
    return Path(ser._read_str(f))

@writer("pathlib.PosixPath")
def write_posix_path(ser, data, out):
    ser._write_str(data.as_posix(), out)

@reader("pathlib.PosixPath")
def read_posix_path(ser, f):
    from pathlib import Path
    return Path(ser._read_str(f))

@writer("pathlib._local.WindowsPath")
def write_local_windows_path(ser, data, out):
    ser._write_str(data.as_posix(), out)

@reader("pathlib._local.WindowsPath")
def read_local_windows_path(ser, f):
    from pathlib import Path
    return Path(ser._read_str(f))

@writer("pathlib.WindowsPath")
def write_windows_path(ser, data, out):
    ser._write_str(data.as_posix(), out)

@reader("pathlib.WindowsPath")
def read_windows_path(ser, f):
    from pathlib import Path
    return Path(ser._read_str(f))

serializer.harvest(globals())
