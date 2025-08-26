from decimal import Decimal

from ...core import writer, reader, read_str, write_str

@writer("decimal.Decimal")
def write_decimal(data, out):
    write_str(str(data), out)

@reader("decimal.Decimal")
def read_decimal(f):
    return Decimal(read_str(f))
