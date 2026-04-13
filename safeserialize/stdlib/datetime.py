from .. import *
import struct

@writer("datetime.datetime")
def write_datetime(ser, data, out):
    ser._write_str(data.isoformat(), out)

@reader("datetime.datetime")
def read_datetime(ser, f):
    from datetime import datetime
    return datetime.fromisoformat(ser._read_str(f))

@writer("datetime.date")
def write_date(ser, data, out):
    ser._write_str(data.isoformat(), out)

@reader("datetime.date")
def read_date(ser, f):
    from datetime import date
    return date.fromisoformat(ser._read_str(f))

@writer("datetime.time")
def write_time(ser, data, out):
    ser._write_str(data.isoformat(), out)

@reader("datetime.time")
def read_time(ser, f):
    from datetime import time
    return time.fromisoformat(ser._read_str(f))

@writer("datetime.timedelta")
def write_timedelta(ser, data, out):
    # Use nanoseconds in case we need more precision in the future
    nanoseconds = data.microseconds * 1000
    seconds = data.seconds
    days = data.days
    assert -999999999 <= days <= 999999999
    assert 0 <= seconds <= 999999999
    assert 0 <= nanoseconds <= 999999999
    out.write(struct.pack("<iII", days, seconds, nanoseconds))

@reader("datetime.timedelta")
def read_timedelta(ser, f):
    from datetime import timedelta
    days, seconds, nanoseconds = struct.unpack("<iII", f.read(3 * 4))
    microseconds = nanoseconds // 1000
    return timedelta(days=days, seconds=seconds, microseconds=microseconds)

serializer.harvest(globals())
