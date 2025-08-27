from pathlib import Path

from ...core import writer, reader, read_str, write_str

"""
On Linux, "a\\b" is a valid file name,
not a directory "a" with a file "b".
We convert to "a/b" with as_posix().
"""

@writer("pathlib._local.PosixPath")
def write_local_posix_path(data, out):
    write_str(data.as_posix(), out)

@reader("pathlib._local.PosixPath")
def read_local_posix_path(f):
    return Path(read_str(f))

@writer("pathlib.PosixPath")
def write_posix_path(data, out):
    write_str(data.as_posix(), out)

@reader("pathlib.PosixPath")
def read_posix_path(f):
    return Path(read_str(f))

@writer("pathlib._local.WindowsPath")
def write_local_windows_path(data, out):
    write_str(data.as_posix(), out)

@reader("pathlib._local.WindowsPath")
def read_local_windows_path(f):
    return Path(read_str(f))

@writer("pathlib.WindowsPath")
def write_windows_path(data, out):
    write_str(data.as_posix(), out)

@reader("pathlib.WindowsPath")
def read_windows_path(f):
    return Path(read_str(f))
