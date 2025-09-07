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

## Contributing

Want to serialize a data type that is not yet supported?
Open an [issue](https://github.com/99991/safeserialize/issues) or make a [pull request](https://github.com/99991/safeserialize/pulls).

## FAQ

* Q: I want to serialize as a string, not as bytes.
* A: No problem! Simply encode the binary data with `base64`:

```python
from safeserialize import dumps, loads
import base64

data = {b"Hello": b"World!"}

serialized_str = base64.b64encode(dumps(data)).decode("ascii")

# The serialized data is a string
assert isinstance(serialized_str, str)

deserialized_data = loads(base64.b64decode(serialized_str))

assert data == deserialized_data

# For brevity, the following wrappers do the same as the code above.
from safeserialize import dump_base64, load_base64

serialized_str = dump_base64(data)

assert data == load_base64(serialized_str)

print("Serialization and deserialization successful!")
```

* Q: The serialized data is too big. How do I make it smaller?
* A: Use compression, for example `zlib` (mature) or `bz2` (high compression, slower). If you are willing to install third-party libraries, `lz4` (less compression, but very fast decompression) or `zstd` (high compression ratio, very high decompression speed) are also an option.

```python
from safeserialize import dumps, loads
import bz2

data = [{b"Hello": b"World!"}] * 100

serialized_bytes = dumps(data)

compressed_bytes = bz2.compress(serialized_bytes)

percent = len(compressed_bytes) * 100 / len(serialized_bytes)

print(f"Compressed to {percent:.1f} % of original size")

decompressed_bytes = bz2.decompress(compressed_bytes)

deserialized_data = loads(decompressed_bytes)
print("Serialization and deserialization successful!")
```
