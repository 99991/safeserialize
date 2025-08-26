import uuid

from ...core import writer, reader, read_bytes, write_bytes

@writer("uuid.UUID")
def write_uuid(data, out):
    write_bytes(data.bytes, out)

@reader("uuid.UUID")
def read_uuid(f):
    return uuid.UUID(bytes=read_bytes(f))
