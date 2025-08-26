import uuid
from pathlib import Path
from decimal import Decimal
from datetime import datetime
from fractions import Fraction
from collections import deque, Counter, OrderedDict

from safeserialize import dumps, loads

def test_stdlib():
    a = datetime(2025, 8, 25, 22, 55, 47, 343776)
    b = datetime.fromtimestamp(0)

    data = {
        "fraction": Fraction(3, 4),
        "decimal": Decimal("3.14159265358979323846"),
        "datetime": a,
        "date": a.date(),
        "time": a.time(),
        "timedelta": a - b,
        "timedelta_negative": b - a,
        "uuid": uuid.uuid5(uuid.NAMESPACE_DNS, b"example.com"),
        "path": Path("some/relative/path"),
        "deque": deque([1, 2, 3]),
        "counter": Counter("banana"),
        "ordered_dict": OrderedDict([("apple", 1), ("banana", 2)]),
    }

    serialized = dumps(data)

    deserialized = loads(serialized)

    assert data == deserialized, f"{data} != {deserialized}"
