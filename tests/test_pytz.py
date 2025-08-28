from safeserialize import dumps, loads
import pytz
from safeserialize.types.pytz import _timezones

def test_pytz():
    for name in _timezones:
        timezone = pytz.timezone(name)
        serialized = dumps(timezone)
        deserialized = loads(serialized)
        assert timezone == deserialized
