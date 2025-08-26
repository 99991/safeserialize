from safeserialize import read, write, loads, dumps

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age

def write_person(person, out):
    write(person.name, out)
    write(person.age, out)

def read_person(f):
    name = read(f)
    age = read(f)
    return Person(name, age)

def test_custom():
    people = [
        Person("Bilbo", 111),
        Person("Gandalf", 2000),
    ]

    # Name must match module hierarchy + class name
    name = "tests.test_custom_type.Person"

    serialized_data = dumps(people, writers={name: write_person})

    loaded_people = loads(serialized_data, readers={name: read_person})

    assert people == loaded_people
