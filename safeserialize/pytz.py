from . import *

@writer('pytz.tzinfo.BaseTzInfo', sub = True)
def write_timezone(ser, tz, out):
    ser._write_str(tz.zone, out)

@reader('pytz.tzinfo.BaseTzInfo')
def read_timezone(ser, f):
    import pytz
    zone = ser._read_str(f)
    return pytz.timezone(zone)

serializer.harvest(globals())
