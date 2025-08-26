from datetime import datetime, date, time, timedelta
import struct

from ...core import writer, reader, read_str, write_str

@writer("datetime.datetime")
def write_datetime(data, out):
    write_str(data.isoformat(), out)

@reader("datetime.datetime")
def read_datetime(f):
    return datetime.fromisoformat(read_str(f))

@writer("datetime.date")
def write_date(data, out):
    write_str(data.isoformat(), out)

@reader("datetime.date")
def read_date(f):
    return date.fromisoformat(read_str(f))

@writer("datetime.time")
def write_time(data, out):
    write_str(data.isoformat(), out)

@reader("datetime.time")
def read_time(f):
    return time.fromisoformat(read_str(f))

@writer("datetime.timedelta")
def write_timedelta(data, out):
    # Use nanoseconds in case we need more precision in the future
    nanoseconds = data.microseconds * 1000
    seconds = data.seconds
    days = data.days
    assert -999999999 <= days <= 999999999
    assert 0 <= seconds <= 999999999
    assert 0 <= nanoseconds <= 999999999
    out.write(struct.pack("<iII", days, seconds, nanoseconds))

@reader("datetime.timedelta")
def read_timedelta(f):
    days, seconds, nanoseconds = struct.unpack("<iII", f.read(3 * 4))
    microseconds = nanoseconds // 1000
    return timedelta(days=days, seconds=seconds, microseconds=microseconds)
