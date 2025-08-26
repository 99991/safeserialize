from collections import deque, Counter, OrderedDict

from ...core import writer, reader, read, write, read_int, write_int

# TODO: defaultdict for some reasonable default types (list, int, ...)?

@writer("collections.deque")
def write_deque(data, out):
    write_int(len(data), out)
    for value in data:
        write(value, out)

@reader("collections.deque")
def read_deque(f):
    length = read_int(f)
    return deque(read(f) for _ in range(length))

@writer("collections.Counter")
def write_counter(data, out):
    write_int(len(data), out)
    for key, value in data.items():
        write(key, out)
        write(value, out)

@reader("collections.Counter")
def read_counter(f):
    length = read_int(f)
    c = Counter()
    for _ in range(length):
        key = read(f)
        value = read(f)
        c[key] = value
    return c

@writer("collections.OrderedDict")
def write_ordered_dict(data, out):
    write_int(len(data), out)
    for key, value in data.items():
        write(key, out)
        write(value, out)

@reader("collections.OrderedDict")
def read_ordered_dict(f):
    length = read_int(f)
    d = OrderedDict()
    for _ in range(length):
        key = read(f)
        value = read(f)
        d[key] = value
    return d
