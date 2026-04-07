from .. import *

@writer("decimal.Decimal")
def write_decimal(ser, data, out):
    ser._write_str(str(data), out)

@reader("decimal.Decimal")
def read_decimal(ser, f):
    from decimal import Decimal
    return Decimal(ser._read_str(f))

serializer.harvest(globals())
