from fractions import Fraction

from ...core import writer, reader, read_int, write_int

@writer("fractions.Fraction")
def write_fraction(data, out):
    write_int(data.numerator, out)
    write_int(data.denominator, out)

@reader("fractions.Fraction")
def read_fraction(f):
    numerator = read_int(f)
    denominator = read_int(f)
    return Fraction(numerator, denominator)
