# SafeSerialize

SafeSerialize is a safe and extensible binary serialization library for Python.

Ever got an error like `TypeError: Object of type set is not JSON serializable`? - No more!

This library supports

- Python's builtin data types (`set`, `frozenset`, `dict`, `bytes`, ...),
- many types from Python's standard library (`datetime`, `decimal`, `Counter`, `deque`, ...),
- NumPy arrays and scalar data types,
- PyTorch tensors,
- SciPy BSR, CSR, CSC and COO sparse matrices,
- (experimental) Pandas support,
- custom user-defined types.

Unlike [`pickle`](https://docs.python.org/3/library/pickle.html),
this library is designed to be safe and does not execute arbitrary code when loading untrusted data.

## Installation

You can install SafeSerialize from PyPI:

```bash
pip install safeserialize
```

Third party libraries (e.g. NumPy) are optional.
Support will automatically be enabled once they are installed.

## Usage

Here is a quick example of how to use SafeSerialize.
It should mostly be a drop-in replacement for `pickle`.

```python
from safeserialize import dumps, loads
from datetime import datetime
from decimal import Decimal
from collections import Counter

# Create a complex object
data = {
    "an_integer": 42,
    "a_string": "Hello, World!",
    "a_list": [1, 2.0, "three"],
    frozenset("a_set"): {"foo", "bar"},
    "a_datetime": datetime.now(),
    "a_counter": Counter("banana"),
    "a_decimal": Decimal("3.14159"),
}

# Serialize the data as a bytes
serialized_bytes = dumps(data)

# Deserialize the object
deserialized_data = loads(serialized_bytes)

assert data == deserialized_data
print("Serialization and deserialization successful!")
```

Serialization directly to files is also supported.

```python
from safeserialize import dump, load

data = {1, 2.0, ..., "four!"}

filename = "data.safeserialize"

with open(filename, "wb") as f:
    dump(data, f)

with open(filename, "rb") as f:
    deserialized_data = load(f)

assert data == deserialized_data
print("Serialization and deserialization successful!")
```

For more usage examples, see the [tests](https://github.com/99991/safeserialize/tree/main/tests).

## Running Tests

To run the tests, first clone the repository and install the development dependencies:

```bash
git clone https://github.com/your-username/safeserialize.git
cd safeserialize
pip install -e .[test]
```

Then, run `pytest` from the root directory:

```bash
pytest
```
