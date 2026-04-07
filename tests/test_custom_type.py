from safeserialize import reader, writer

class ForeignPerson:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age

@writer(ForeignPerson)
def write_person(self, person, out):
    self._write(person.name, out)
    self._write(person.age, out)

@reader(ForeignPerson)
def read_person(self, f):
    name = self._read(f)
    age = self._read(f)
    return ForeignPerson(name, age)

def test_custom_foreign():

    from safeserialize import dumps, loads, Serializer

    people = [
        ForeignPerson("Bilbo", 111),
        ForeignPerson("Gandalf", 2000),
    ]

    # Name must match module hierarchy + class name
    name = "tests.test_custom_type.Person"

    serializer = Serializer(globals())
    
    serialized_data = serializer.dumps(people)

    loaded_people = serializer.loads(serialized_data)

    assert people == loaded_people


class OurPerson:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age

    def __safeserialize__(self, ser, out):
        ser._write(self.name, out)
        ser._write(self.age, out)

    def __safedeserialize__(cls, ser, f):
        name = ser._read(f)
        age = ser._read(f)
        return cls(name, age)

def test_custom_ours():

    from safeserialize import dumps, loads, Serializer

    people = [
        OurPerson("Bilbo", 111),
        OurPerson("Gandalf", 2000),
    ]
    
    serializer = Serializer([OurPerson])
    
    serialized_data = serializer.dumps(people)

    loaded_people = serializer.loads(serialized_data)

    assert people == loaded_people
