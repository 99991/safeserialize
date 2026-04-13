from .. import *

@writer("uuid.UUID")
def write_uuid(ser, data, out):
    ser._write_bytes(data.bytes, out)

@reader("uuid.UUID")
def read_uuid(ser, f):
    import uuid
    return uuid.UUID(bytes=ser._read_bytes(f))

serializer.harvest(globals())
