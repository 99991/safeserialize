from .. import *

# TODO: defaultdict for some reasonable default types (list, int, ...)?

@writer("collections.deque")
def write_deque(ser, data, out):
    ser._write_int(len(data), out)
    for value in data:
        ser._write(value, out)

@reader("collections.deque")
def read_deque(ser, f):
    from collections import deque
    length = ser._read_int(f)
    return deque(ser._read(f) for _ in range(length))

@writer("collections.Counter")
def write_counter(ser, data, out):
    ser._write_int(len(data), out)
    for key, value in data.items():
        ser._write(key, out)
        ser._write(value, out)

@reader("collections.Counter")
def read_counter(ser, f):
    from collections import Counter
    length = ser._read_int(f)
    c = Counter()
    for _ in range(length):
        key = ser._read(f)
        value = ser._read(f)
        c[key] = value
    return c

@writer("collections.OrderedDict")
def write_ordered_dict(ser, data, out):
    ser._write_int(len(data), out)
    for key, value in data.items():
        ser._write(key, out)
        ser._write(value, out)

@reader("collections.OrderedDict")
def read_ordered_dict(ser, f):
    from collections import OrderedDict
    length = ser._read_int(f)
    d = OrderedDict()
    for _ in range(length):
        key = ser._read(f)
        value = ser._read(f)
        d[key] = value
    return d

serializer.harvest(globals())
