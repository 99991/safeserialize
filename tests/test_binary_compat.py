import pytest
from pathlib import Path

"""Test for binary compatibility of dumps across versions of the package.

Any object sent for a roundtrip is recorded in `roundtrip.DATA`, along with the
comparison method which must be used to test equality (and the test file and
function which produced it).

`test_dump` dumps `roundtrip.DATA` into `curr.bin`.  The idea is to commit this
dump into the repo as `dump.bin`.  In the next version of the package, the
contents of `dump.bin` are compared to the current dump:

- `test_byte_compare_dumps` compares the dumps on the binary level. Currently,
  this may fail because `(frozen)set` does not guarantee order, so the test is
  skipped.

- `test_compare_dump_to_data` loads `dump.bin` into an object and compares it
  to `roundtrip.DATA`.

- `test_compare_dumps` loads both `dump.bin` and `curr.bin` into objects and
  compares the objects.

If object comparison fails, the assertion provides the information on the
origin of the object: the test file, the test function and the roundtrip number
within the function.

This test module must be executed last, which is achieved using pytest plugin
`pytest-order`.

"""

pytestmark = pytest.mark.order(-1)

ifdump = pytest.mark.skipif(
    not Path("dump.bin").exists(),
    reason = "dump.bin does not exist"
)

@pytest.mark.order(-2)
def test_dump():
    from .roundtrip import DATA
    from safeserialize import dump
    with open('curr.bin', 'wb') as f:
        dump(dict(DATA), f)

@pytest.mark.skip("mismatch due to frozenset in test_builtins")
@ifdump
def test_byte_compare_dumps():
    with open('dump.bin', 'rb') as f:
        dump = f.read()
    with open('curr.bin', 'rb') as f:
        curr = f.read()
    assert dump == curr

@ifdump
def test_compare_dump_to_data():
    from safeserialize import load
    with open('dump.bin', 'rb') as f:
        deserialized_dump = load(f)
    from .roundtrip import DATA
    compare(deserialized_dump, DATA)
    
@ifdump
def test_compare_dumps():
    from safeserialize import load
    with open('dump.bin', 'rb') as f:
        deserialized_dump = load(f)
    with open('curr.bin', 'rb') as f:
        deserialized_curr = load(f)
    compare(deserialized_dump, deserialized_curr)

def compare(prev, curr):
    from .roundtrip import CMP_TYPES
    for func, curr_list in curr.items():
        prev_list = prev[func]
        for n, (prev_item, curr_item) in enumerate(zip(prev_list, curr_list)):
            assert prev_item[0] == curr_item[0] # comparision types
            assert_f = CMP_TYPES[prev_item[0]]
            try:
                assert_f(prev_item[1], curr_item[1])
            except AssertionError as err:
                raise AssertionError(f'{func}[{n}]: {err}')
