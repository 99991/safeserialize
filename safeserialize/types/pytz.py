from ..core import iwriter, reader, write_str, read_str
import pytz

@iwriter(pytz.tzinfo.BaseTzInfo)
def write_timezone(tz, out):
    write_str(tz.zone, out)

@reader('pytz.tzinfo.BaseTzInfo')
def read_timezone(f):
    zone = read_str(f)
    return pytz.timezone(zone)
