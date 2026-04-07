import pytest
from collections import deque

def roundtrip(value, serializer):
    serialized_value = serializer.dumps(value)
    print(serialized_value)
    roundtrip_value = serializer.loads(serialized_value)
    assert roundtrip_value == value, f"{roundtrip_value} != {value}"
    return serialized_value

def test_import_type_module():

    from safeserialize import serializer, dumps, loads
    with pytest.raises(TypeError):
        serializer.dumps(deque([1,2,3]))
    with pytest.raises(TypeError):
        dumps(deque([1,2,3]))

    import safeserialize.stdlib.collections
    roundtrip(deque([1,2,3]), serializer)

    value = deque([1,2,3])
    roundtrip_value = loads(dumps(value))
    assert roundtrip_value == value, f"{roundtrip_value} != {value}"
    
def test_harvest_from_module():

    from safeserialize import Serializer
    import safeserialize.stdlib.collections

    serializer = Serializer()
    with pytest.raises(TypeError):
        roundtrip(deque([1,2,3]), serializer)

    serializer.harvest(safeserialize.stdlib.collections)
    roundtrip(deque([1,2,3]), serializer)
    
def test_harvest_from_class():

    from safeserialize import Serializer, reader, writer
    
    class Functions:

        @writer("collections.deque")
        def write_deque(ser, data, out):
            ser._write_int(len(data), out)
            for value in data:
                ser._write(value, out)

        @reader("collections.deque")
        def read_deque(ser, f):
            length = ser._read_int(f)
            return deque(ser._read(f) for _ in range(length))

    with pytest.raises(TypeError):
        roundtrip(deque([1,2,3]), Serializer())
    roundtrip(deque([1,2,3]), Serializer(Functions))

def test_harvest_from_list():

    from safeserialize import Serializer, reader, writer
    
    @writer("collections.deque")
    def write_deque(ser, data, out):
        ser._write_int(len(data), out)
        for value in data:
            ser._write(value, out)

    @reader("collections.deque")
    def read_deque(ser, f):
        length = ser._read_int(f)
        return deque(ser._read(f) for _ in range(length))

    with pytest.raises(TypeError):
        roundtrip(deque([1,2,3]), Serializer())
    roundtrip(deque([1,2,3]), Serializer([write_deque, read_deque]))

    
def test_special_method():

    class Foo:
        
        def __init__(self, x = None, y = None):
            self.x = x
            self.y = y
            
        def __eq__(self, other):
            return self.x == other.x and self.y == other.y

    class SafelySerializableFoo(Foo):
        
        def __safeserialize__(self, ser, out):
            ser._write(self.x, out)
            ser._write(self.y, out)

        def __safedeserialize__(cls, ser, f):
            return cls(ser._read(f), ser._read(f))

    from safeserialize import Serializer
        
    with pytest.raises(TypeError):
        roundtrip(Foo(42,24), Serializer())
    with pytest.raises(TypeError):
        roundtrip(Foo(42,24), Serializer([Foo]))
    with pytest.raises(TypeError):
        roundtrip(Foo(42,24), Serializer([SafelySerializableFoo]))
        
    roundtrip(SafelySerializableFoo(42,24), Serializer([SafelySerializableFoo]))

def test_blank_serializer():

    from safeserialize import Serializer

    with pytest.raises(TypeError):
        roundtrip(42, Serializer(blank = True))

    with pytest.raises(TypeError):
        roundtrip("foo", Serializer(blank = True))

    roundtrip(42, Serializer())
    roundtrip("foo", Serializer())

def test_override_builtin():

    from safeserialize import Serializer, reader, writer
    
    original_serialized_42 = roundtrip(42, Serializer(header = False))

    @writer('builtins.int', 2)
    def write_int(ser, data, out):
        out.write(str(data).encode() + b'X')
    @reader('builtins.int', 2)
    def read_int(ser, f):
        number_bytes = b''
        while (digit := f.read(1)) != b'X':
            print(digit)
            number_bytes += digit
            if not digit:
                raise ValueError()
        return int(number_bytes.decode())

    blank_serialized_42 = roundtrip(
        42, Serializer([write_int, read_int], header = False, blank = True))
    overriden_serialized_42 = roundtrip(
        42, Serializer([write_int, read_int], header = False))

    assert blank_serialized_42 == b'\x0242X'
    assert original_serialized_42 != blank_serialized_42
    #assert blank_serialized_42 == overriden_serialized_42
