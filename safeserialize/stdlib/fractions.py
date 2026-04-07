from .. import *

@writer("fractions.Fraction")
def write_fraction(ser, data, out):
    ser._write_int(data.numerator, out)
    ser._write_int(data.denominator, out)

@reader("fractions.Fraction")
def read_fraction(ser, f):
    from fractions import Fraction
    numerator = ser._read_int(f)
    denominator = ser._read_int(f)
    return Fraction(numerator, denominator)

serializer.harvest(globals())
