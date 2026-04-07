import pytz
from .roundtrip import *

def test_pytz():
    from safeserialize import dumps, loads
    for tz_name in pytz.all_timezones:
        roundtrip(pytz.timezone(tz_name))
    roundtrip(pytz.UTC)
