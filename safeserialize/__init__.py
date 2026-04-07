"""A safe and extensible binary serialization library for Python.

Ever got an error like `TypeError: Object of type set is not JSON serializable`? - No more!

This library supports

- Python's builtin data types (`set`, `frozenset`, `dict`, `bytes`, ...),
- many types from Python's standard library (`datetime`, `decimal`, `Counter`, `deque`, ...),
- NumPy arrays and scalar data types,
- PyTorch tensors,
- SciPy BSR, CSR, CSC and COO sparse matrices,
- (experimental) Pandas support,
- custom user-defined types.

Unlike `pickle`, this library is designed to be safe and does not execute arbitrary code
when loading untrusted data.


BASIC USAGE

Here is a quick example of how to use SafeSerialize.  It should mostly be a drop-in
replacement for `pickle`.

    from safeserialize import dumps, loads

    # Create a complex object
    data = {
        "an_integer": 42,
        "a_string": "Hello, World!",
        "a_list": [1, 2.0, "three"],
        frozenset("a_set"): {"foo", "bar"},
    }

    # Serialize the data as a bytes
    serialized_bytes = dumps(data)

    # Deserialize the object
    deserialized_data = loads(serialized_bytes)

    assert data == deserialized_data
    print("Serialization and deserialization successful!")

Serialization directly to files is also supported.

    from safeserialize import dump, load

    data = {1, 2.0, ..., "four!"}

    filename = "data.safeserialize"

    with open(filename, "wb") as f:
        dump(data, f)

    with open(filename, "rb") as f:
        deserialized_data = load(f)

    assert data == deserialized_data
    print("Serialization and deserialization successful!")


(DE)SERIALIZING SUPPORTED DATA TYPES OTHER THAN BUILTINS

The quickest way to load support for all datatypes offered by safeserialize:

    from safeserialize.all import dumps, loads

No harm is done by loading all the support (other than perhaps keeping several small
unused functions in memory).  In particular, safeserialize will not attempt to load numpy,
scipy etc. if you `import safeserialize.all`.

More control is available by loading individual datatype modules.  For example:

    from safeserialize.numpy import dumps, loads
    import numpy as np

    data = np.array([1, 2, 3])
    serialized = dumps(data)
    deserialized = loads(serialized)
    assert np.array_equal(data, deserialized)

Importing the datatype submodule later is also fine, and safeserialize (and its datatype
submodules) may be loaded before or after the library they support.  (Note, however, that
in principle, all datatype submodules should be loaded before the first serialization
takes place.  See argument `sub` of decorator `writer` for the gory details.)

    import numpy as np

    from safeserialize import dumps, loads
    import safeserialize.numpy

    ...

Currently, the following datatype submodules are offered:

    import safeserialize.builtins           # loaded automatically
    import safeserialize.stdlib.collections
    import safeserialize.stdlib.datetime
    import safeserialize.stdlib.decimal
    import safeserialize.stdlib.fractions
    import safeserialize.stdlib.pathlib
    import safeserialize.stdlib.uuid
    import safeserialize.stdlib.all # load all supporting submodules in stdlib
    import safeserialize.numpy
    import safeserialize.pandas
    import safeserialize.pytz
    import safeserialize.scipy
    import safeserialize.torch
    import safeserialize.all        # load all supporting submodules


FINE CONTROL OVER WHICH DATA TYPES TO SERIALIZE

Functions `loads`, `dumps` etc. we have been using above are merely convenient
abbreviations for methods `loads`, `dumps` etc. of the default serializer --- an object
instantiated in the `safeserialize` module and accessible as `serializer`.

    from safeserialize import serializer, dump, dumps, load, loads

    assert dump  == serializer.dump
    assert dumps == serializer.dumps
    assert load  == serializer.load
    assert loads == serializer.loads

Importing a datatype submodule automatically registers all data types supported by the
module with the default serializer.  The default serializer is an instance of class
`Serializer`.  Creating other instances of this class, we can gain finer control over what
data types may get serialized in which part of the program.

For example, having imported numpy and stdlib.datetime support below, the default
serializer (underlying plain `dumps`) will serialize both numpy array `a` and datetime
`d`.  Serializer `ser`, on the other hand, will only serialize the datetime, as it was
only initialized with the `stdlib.datetime` submodule as an argument.

    from safeserialize import dumps, loads, Serializer
    import safeserialize.numpy, safeserialize.stdlib.datetime
    import numpy as np, datetime

    ser = Serializer(safeserialize.stdlib.datetime)

    a = np.array([1, 2, 3])
    d = datetime.datetime.now()

    dumps(d) # ok
    dumps(a) # ok

    ser.dumps(d) # ok
    ser.dumps(a) # raises TypeError

The `Serializer` constructor may be given any number of modules, and in fact not only
modules --- read on, and see the documentation of the class.


DEFINING (DE)SERIALIZERS FOR OTHER DATA TYPES

To support (de)serialization of a data type, two functions must be defined: a writer (for
serialization) and a reader (for deserialization).  The functions must be decorated with
decorators `writer` and `reader`, and then registered with a `Serializer` (either default
or custom).  Before delving into the details of how to define and decorate the readers and
writers, we present a complete example illustrating a convenient way of registering all
reader/writer functions defined (and marked) in a module by "harvesting" the module's
globals.  (Type `datetime` is already supported by `safeserialize`, the code below is an
almost verbatim copy of the actual code in the module `safeserialize.stdlib.datetime`.)

    import datetime
    from safeserialize import serializer, reader, writer, loads, dumps

    @writer("datetime.datetime")
    def write_datetime(ser, data, f):
        ser._write_str(data.isoformat(), f)

    @reader("datetime.datetime")
    def read_datetime(ser, f):
        from datetime import datetime
        return datetime.fromisoformat(ser._read_str(f))

    serializer.harvest(globals())

A writer function should take three arguments: a `Serializer` instance `ser`, the `data`
to be serialized, and the output stream `f` (any instance of `io.IOBase`); there is no
return value.  A writer can use `f.write` to write directly to the output stream, but it
can also deploy existing writer functions to write out more "primitive" data types.  To
write out an object of an unknown type, use `ser._write` (see the `dict` example below),
and to write out an object of a known type, use a shorthand method (`ser._write_...`),
e.g. as shown above, `ser._write_str` writes out a string.

A reader function should take two arguments: a `Serializer` instance `ser`, and the output
stream `f` (any instance of `io.IOBase`), and return deserialized data.  A reader can use
`f.read` to read directly from the input stream, but it can also deploy existing reader
functions to read in more "primitive" data types.  To read in an object of an unknown
type, use `ser._read` (see the `dict` example below), and to read in an object of a known
type, use a shorthand method (`ser._read_...`), e.g. as shown above, `ser._read_str` reads
in a string.

Decorator functions `reader` and `writer` should be given the data type as the argument.
The data type may be given as a "fully qualified name" (`<module name>.<qualified class
name>`), as above.  This makes it possible to define a reader/writer without importing the
module defining the datatype (above, `datetime`); note that the datetime reader above,
which has to actually instantiate the datetime, imports it within the function.

Decorator functions may also be given the data type as an actual type object (see below).
Furthermore, the data type may also be given a numeric type id (in the range of 0-254).
This may be used to shorten the serialized data, as it avoids the need to spell out the
name of the data type in the serialization.  All builtin data types supported by
`safeserialize` have a type id.  (Use `Serializer.info()` to see the currently assigned
type ids.) For example, here is how the `dict` type support is implemented, also
showcasing the usage of `ser._write` and `ser._read` mentioned above.

    @writer(dict, 25)
    def write_dict(ser, data, out):
        ser._write_int(len(data), out)
        for key, value in data.items():
            ser._write(key, out)
            ser._write(value, out)

    @reader(dict, 25)
    def read_dict(ser, f):
        length = ser._read_int(f)
        return {ser._read(f): ser._read(f) for _ in range(length)}

For further details, see the documentation of decorator functions `reader` and `writer`.

Finally, in user-defined classes (de)seriazation can also be implemented within the class
itself, by defining special methods `__safeserialize__` (the writer) and
`__safedeserialize__` (the reader).  These methods should be defined in the same fashion
as the decorated functions above, with two exceptions.  First, they should *not* be
decorated using `reader` and `writer`.  Second, their argument structure is a bit
different.

The `__safeserialize__` method takes the same argument as a decorated writer function, but
the `ser`ializer and the `data` argument are swapped, as the latter is actually the
`self`.  The `__safedeserialize__` method takes one additional argument, the class, which
becomes the first argument; note that despite this, `__safedeserialize__` should not be
decorated as a `classmethod`.

Finally, the class should be registered with a serializer, either by harvesting as above,
or by using method `register` as shown below.

    from safeserialize import serializer

    class Foo:
        
        def __init__(self, x = None, y = None):
            self.x = x
            self.y = y
            
        def __eq__(self, other):
            return self.x == other.x and self.y == other.y

        def __safeserialize__(self, ser, out):
            ser._write(self.x, out)
            ser._write(self.y, out)

        def __safedeserialize__(cls, ser, f):
            return cls(ser._read(f), ser._read(f))

    serializer.register(Foo)
    
    foo = Foo(42,24)
    assert loads(dumps(foo)) == foo

"""

import io
import base64
import struct
import importlib
import sys
from collections import namedtuple, defaultdict
from functools import partial

# todo: what sort of object may the version be? Perhaps it makes sense to allow
# for strings as well.
VERSION = 2
# Adapted from PNG file signature
# https://www.libpng.org/pub/png/spec/1.2/PNG-Rationale.html#R.PNG-file-signature
FILE_SIGNATURE = b'\x89SER\r\n\x1a\n'

# This is the only type id fixed by the core.
TYPE_CUSTOM = 255


def writer(typ: type | str, type_id: int | None = None,
           *, raw: bool = False, sub: bool = False):
    
    """Return a decorator marking the function as a serializer for the given type.

    Arguments:

    - typ: The type of data to be serialized by the decorated function.

        The type may be given either as a type object or as a "fully qualified name":
        `cls.__module__ + '.' + cls.__qualname__`, i.e. a string consisting of the name of
        the module where the type `cls` is defined plus the qualified name of the type,
        separated by a period.  The latter is perfect for defining a writer function
        without having to import the module containing the type.

    - type_id: The numeric type id used to represent type `typ` in serialized data.

        If the `type_id` is `None`, the type id of serialized data will be `TYPE_CUSTOM`,
        with the custom subtype (a string) determined by `typ`.

    - raw: A True value signals that the decorated function is a raw writer.

        Normally, the type_id is written into the serialization automatically.  A raw
        writer function, however, is responsible for writing out the type_id on its own.
        This is an optimization used for builtin types `bool` and `int` in submodule
        `builtins`.

    - sub: A True value signals that the decorated function handles subclasses of the
           given type, as well.

        The "subclass" actually refers to any descendant class of the type,
        i.e. `issubclass` is used to check whether the data should be serialized by the
        decorated function.

        Clearly, `issubclass` cannot be used without a real type object.  However, we
        don't want to necessarily import the module defining `typ` when defining the
        serializer.  Therefore, the following mechanism is set up.  If `typ` is given as a
        string, safeserialize attempts to instantiate the corresponding type object upon
        the first serialization: this succeeds if the module defining the type is loaded;
        if it fails, the sub writer is thrown away.  See submodule `pytz` for an example
        of use.

    The decorated function should have the following signature:

        Callable[ [ser: Serializer, data: Any, out: io.IOBase], None ]

    To be actually used for serialization, the marked function must be registered into a
    `Serializer`.

    Another way of declaring a reader function is by defining the special method
    `__safedeserialize__` on the class whose instances are to be deserialized.

    See the module help for a tutorial on how to define readers and writers.

    """

    assert not (raw and sub), "A writer cannot be declared both raw and sub."
    purpose = '_rwriters' if raw else '_swriters' if sub else '_writers'
    def decorator(func):
        func._safeserialize = _funcInfo(purpose, typ, type_id)
        return func
    return decorator


def reader(typ: type | str | None = None, type_id: int | None = None):
    
    """Return a decorator marking the function as a deserializer for the given type.

    Arguments:

    - typ: The type of data to be deserialized by the decorated function.

        The type may be given either as a type object or as a "fully qualified name":
        `cls.__module__ + '.' + cls.__qualname__`, i.e. a string consisting of the name of
        the module where the type `cls` is defined plus the qualified name of the type,
        separated by a period.  The latter is perfect for defining a reader function
        without having to import the module containing the type.  `typ=None` only makes
        sense if the reader corresponds to a raw writer, see the definitions of readers
        and writers for builtin types `bool` and `int` in submodule `builtins`.

    - type_id: The numeric type id used to represent type `typ` in serialized data.

        If `type_id` is `None`, the data is assumed to have been serialized with type id
        `TYPE_CUSTOM`, with the custom subtype (a string) determined by `typ`.

    The decorated function should have the following signature:

        Callable[ [ser: Serializer, f: io.IOBase], Any ]

    To be actually used for deserialization, the marked function must be registered into a
    `Serializer`.

    Another way of declaring a reader function is by defining the special method
    `__safedeserialize__` on the class whose instances are to be deserialized.

    See the module help for a tutorial on how to define readers and writers.
    

    """
    
    def decorator(func):
        func._safeserialize = _funcInfo('_readers', typ, type_id)
        return func
    return decorator


_funcInfo = namedtuple('_fInfo', 'purpose,type,type_id')


class Serializer:

    """Create a new (de)serializer object.

    An instance of this class encapsulates a collection of reader and writer functions
    responsible for (de)serialization of individual data types.  Once the readers and
    writers are registered (via the constructor, or methods `harvest` and `register`),
    serialization and deserialization proceeds via methods `dump`, `dumps`, `load` and
    `loads`.  The class also facilitates the invocation of one reader/writer function from
    another (see method `__getattr__`).

    Arguments:

    - sources: The containers to be harvested for readers and writers.  See method
               `.harvest` for details.

    - header: Controls whether the serialized data should be prefixed by the safeserialize
              file signature.  This setting can be overridden for individual `.dump`s and
              `.load`s.

    - blank: Normally, a new `Serializer` object is automatically populated by readers and
             writers for builtin data types (by harvesting submodule `builtins`). Setting
             `blank=True` prevents this, building the serializer from a blank slate.

    """
    
    def __init__(self, *sources, header: bool = True, blank: bool = False):
        
        self._header = header
        
        self._readers = {}
        self._writers = {}
        self._rwriters = {}
        self._swriters = {}
        self._type_ids = {} # string type --> int type id

        if not blank:
            from . import builtins
            self.harvest(builtins)
            
        self.harvest(*sources)

    def harvest(self, *sources):
        """Register the reader and writer functions found in the `sources`.

        A source may be an object with a `__dict__` attribute (like a module), an iterable
        (like a list) or another `Serializer`.

        This method will attempt to `.register` all items in the object's `__dict__` or
        the iterable.  See the documentation of method `register` for details on what
        counts as a registerable item.

        """
        
        for source in sources:
            if isinstance(source, Serializer):
                pass # todo
            else:
                try:
                    src = source.__dict__.values()
                except:
                    try:
                        src = source.values()
                    except:
                        src = source
                try:
                    iterator = iter(src)
                except TypeError:
                    raise TypeError(
                        f'{source} is not a valid source of readers/writers.'
                    ) from None

                for func_or_cls in iterator:
                    self.register(func_or_cls)

    def register(self, func_or_cls):
        """Register a reader function, a writer function, or a (de)serializable class.

        A reader/writer function submitted to this method should have been marked as a
        reader/writer using decorator `reader`/`writer`.

        A (de)serializable class is a class containing methods `__safeserialize__` and
        `__safedeserialize__`.

        See the module help for further information.

        """
        
        if hasattr(func_or_cls, '_safeserialize'):
            self._register_func(func_or_cls)
        elif hasattr(func_or_cls, '__safeserialize__') and \
             hasattr(func_or_cls, '__safedeserialize__'):
            self._register_cls(func_or_cls)

    def _register_func(self, func):
        info = func._safeserialize
        
        # Convert info.type to a string if it was given as an actual type.
        type_str = info.type.__module__ + '.' + info.type.__qualname__ \
            if isinstance(info.type, type) else info.type

        # Assign the function to the type.  The type may only be None for the
        # readers corresponding to the raw writers (int, bool).  For writers,
        # assign to the actual type rather than the string, if the type was
        # given this way.
        assert info.purpose == '_readers' or type_str is not None
        if type_str is not None:
            if info.purpose == '_swriters':
                getattr(self, info.purpose)[info.type] = func
            else:
                getattr(self, info.purpose)[type_str] = func

        # For readers, assign function to the numeric type_id.
        if info.purpose == '_readers' and info.type_id is not None:
            getattr(self, info.purpose)[info.type_id] = func

        # Populate self._type_ids
        if type_str and info.type_id is not None:
            try:
                assert self._type_ids[type_str] == info.type_id, \
                    f"The reader and the writer for {type_str} declared different type_ids."
            except KeyError:
                self._type_ids[type_str] = info.type_id

        # For writers/swriters, remove the entries in rwriters/(rwriters and
        # writers) to allow redefinitions.
        if info.purpose == '_writers':
            getattr(self, '_rwriters').pop(type_str, None)
        elif info.purpose == '_swriters':
            getattr(self, '_rwriters').pop(type_str, None)
            getattr(self, '_writers').pop(type_str, None)
        
    def _register_cls(self, cls):
        cls_str = cls.__module__ + '.' + cls.__qualname__
        def __safeserialize__(ser, data, out):
            return cls.__safeserialize__(data, ser, out)
        def __safedeserialize__(ser, f):
            return cls.__safedeserialize__(cls, ser, f)
        self._writers[cls_str] = __safeserialize__
        self._readers[cls_str] = __safedeserialize__

    def __getattr__(self, name):
        """Facilitates access to readers and writers from other readers and writers.

        For example, the `list` reader can refer to the `int` reader by `ser._read_int`
        instead of `ser._readers['builtins.int']`, where the `int` reader is stored
        internally.  Some further examples of such "shorthands":
        
        - `self._readers['collections.deque']` --> `self._read_collections_deque`
        
        - `self._writers['builtins.tuple']` --> `self._read_builtins_tuple`
        
        - `self._rwriters['builtins.int']` --> `self._read_int`
        
        - `self._swriters['builtins.int']` --> `self._read_BaseException`

        As exemplified above:
        
        - Dots (`.`) in the dictionary key are converted to underscores.
        
        - `builtins_` may be omitted (or not).
        
        - It does not matter whether a writer comes from `._rwriters`, `._writers`
          or `._swriters`, access is the same.

        Shorthands are generated on the fly, and cached.  Therefore, the overall
        performance impact is negligible even though generating the shorthand is somewhat
        expensive.

        """
        
        if name.startswith('_read_'):
            self._make_shorthand(name, name[6:], [self._readers])
            return getattr(self, name)
        elif name.startswith('_write_'):
            self._make_shorthand(
                name, name[7:], [self._rwriters, self._writers, self._swriters])
            return getattr(self, name)
        else:
            raise AttributeError()

    def _make_shorthand(self, full_name, name, dicts):

        shorthand_dicts = [ self._dotted_names_to_shorthands(dct) for dct in dicts ]
        
        for dct in shorthand_dicts:
            if (func := dct.get(name, None)):
                setattr(self, full_name, partial(func, self))
                break
        else:
            raise AttributeError()

    @staticmethod
    def _dotted_names_to_shorthands(dct):
        return {
            key.replace(".", "_"): value
            for key, value in dct.items()
            if isinstance(key, str)
        } | {
            key[9:].replace(".", "_"): value
            for key, value in dct.items()
            if isinstance(key, str) and key.startswith("builtins.") 
        }

    def info(self, csv: bool = False) -> str:
        """Returns information about registered readers and writers.

        By default, the method generates formatted output suitable for displaying on the
        terminal.
        
        Arguments:

        - csv: When `True`, generate the output in the `.csv` format.

        """
        
        # gather data
        info = defaultdict(lambda: dict(reader = '', writer = ''))
        for t,f in self._readers.items():
            info[t]['reader'] = f.__name__
        for t,f in self._rwriters.items():
            info[t]['writer'] = f.__name__ + ' (raw)'
        for t,f in self._writers.items():
            info[t]['writer'] = f.__name__
        for t,f in self._swriters.items():
            if isinstance(t, type):
                t_str = t.__module__ + '.' + t.__qualname__
                info[t_str]['writer'] = f.__name__ + ' (sub)'
            else:
                info[t]['writer'] = f.__name__

        # merge type and type_id info
        for t,type_id in self._type_ids.items():
            t_str = t.__module__ + '.' + t.__qualname__ if isinstance(t, type) else t
            if (info_t := info.get(t_str, None)) and (info_id := info.get(type_id, None)):
                for rw in ('reader', 'writer'):
                    assert bool(info_t[rw]) != bool(info_id[rw]) or info_t[rw] == info_id[rw]
                del info[t_str]
                del info[type_id]
                info[(type_id, t_str)] = dict(
                    reader = info_t['reader'] or info_id['reader'],
                    writer = info_t['writer'] or info_id['writer'])

        # reformat non-merged info into the merged format
        solo_ids = [ k for k in info if isinstance(k, int)]
        solo_types = [ k for k in info if isinstance(k, str)]
        for type_id in solo_ids:
            info[(type_id, '')] = info.pop(type_id)
        for t in solo_types:
            info[('', t)] = info.pop(t)

        # sort
        rows = [ (k[0], k[1], v['writer'], v['reader']) for k,v in info.items()]
        rows.sort(key = lambda row: (
            (row[0] if isinstance(row[0],int) else TYPE_CUSTOM,
             (0,row[1]) if row[1].startswith('builtins.') else (1,row[1])
             )))

        # format
        rows = [ tuple(str(cell) for cell in row) for row in rows ]
        return self._format_info(rows, csv) if rows else ''

    def _format_info(self, rows, csv: bool):
        if csv:
            return "\n".join(",".join(row) for row in rows)
        else:
            col_lengths = [
                max( len(row[i]) for row in rows )
                for i in range(len(rows[0])) ]
            return "\n".join(
                "  ".join(format(row[i], str(col_lengths[i])) for i in range(len(row)))
                for row in rows)
        
    def dump(self, obj: Any, f: io.IOBase, header: bool = None, version: int = VERSION):
        """Serialize a data object, writing the resulting bytes to output stream.

        Arguments:

        - obj: the object to be serialized

        - f: the output stream (e.g. a file)

        - header: should the serialized data be prefixed by a header?  The default is the
          value of `header` given to `__init__`.

        - version: the version of the serialization format, to be included in the header

        """

        header = self._header if header is None else header
        if header:
            f.write(FILE_SIGNATURE)
            f.write(struct.pack("<I", version))

        # In case of an error, return to the starting position if possible.
        if f.seekable():
            pos = f.tell()
            try:
                self._write(obj, f)
            except:
                f.seek(pos)
                raise
        else:
            self._write(obj, f)

    def _write(self, data, out: io.IOBase):
        self._instantiate_swriter_classes()
        self._write_(data, out)
            
    def _write_(self, data, out: io.IOBase):

        data_type = type(data)
        cls = data.__class__
        data_type_str = cls.__module__ + '.' + cls.__qualname__
        
        try:
            writer = self._rwriters[data_type_str]
        except KeyError:
            pass
        else:
            writer(self, data, out)
            return

        try:
            writer = self._writers[data_type_str]
        except KeyError:
            for dt in self._swriters:
                if isinstance(dt, type) and issubclass(data_type, dt):
                    writer = self._swriters[dt]
                    data_type_str = f"{dt.__module__}.{dt.__name__}"
                    break
            else:
                raise TypeError(f"Writer not implemented for {data_type}") from None

        try:
            type_id = self._type_ids[data_type_str]
        except KeyError:
            out.write(bytes([TYPE_CUSTOM]))
            self._write_str(data_type_str, out)
        else:
            out.write(bytes([type_id]))

        writer(self, data, out)
        
    def _instantiate_swriter_classes(self):
        self._write = self._write_
        swriters = self._swriters
        self._swriters = {}
        for dt in swriters:
            if isinstance(dt, str):
                try:
                    cls = self._instantiate(dt)
                    assert isinstance(cls, type)
                    self._swriters[cls] = swriters[dt]
                except RuntimeError:
                    pass
            else:
                self._swriters[dt] = swriters[dt]

    @staticmethod
    def _instantiate(full_name):
        module_components = full_name.split(".")
        cls_components = [module_components.pop(-1)]
        while module_components:
            try:
                cls = sys.modules[".".join(module_components)]
                for component in cls_components:
                    cls = getattr(cls, component)
                return cls
            except KeyError, AttributeError:
                cls_components = [module_components.pop(-1)]
        raise RuntimeError()
        
    def load(self, f: io.IOBase, header: bool = None, version: int = VERSION) -> Any:
        """Read bytes from the input stream and deserialize them into an object.
        
        Arguments:

        - f: the input stream (e.g. a file)

        - header: is the serialized data be prefixed by a header?  The default is the
          value of `header` given to `__init__`.

        - version: the version of the serialization format to be expected in the header

        """

        header = self._header if header is None else header
        if header:
            signature = f.read(len(FILE_SIGNATURE))
            if signature != FILE_SIGNATURE:
                raise ValueError(f"Invalid file signature {repr(signature)}")
            version_in_header, = struct.unpack("<I", f.read(4))
            assert version_in_header == version

        # In case of an error, return to the starting position if possible.
        if f.seekable():
            pos = f.tell()
            try:
                return self._read(f)
            except:
                f.seek(pos)
                raise
        else:
            return self._read(f)

    def _read(self, f):
        
        type_id, = f.read(1)

        try:
            reader = self._readers[type_id]
        except KeyError:
            if type_id == TYPE_CUSTOM:
                data_type_str = self._read_str(f)
                try:
                    reader = self._readers[data_type_str]
                except KeyError:
                    reader = None
            else:
                reader = None
                
        if reader:
            return reader(self, f)
        elif data_type_str:
            raise TypeError(f"Reader not implemented for type {data_type_str}") from None
        else:
            raise TypeError(f"Reader not implemented for type_id {type_id}") from None

    def dumps(self, obj: Any, header: bool = None, version: int = VERSION) -> bytes:
        """Serialize an object to bytes.

        See `dump` for the meaning of arguments.

        """
        out = io.BytesIO()
        self.dump(obj, out, header = header)
        return out.getvalue()

    def loads(self, data: bytes, header: bool = None, version: int = VERSION):
        """Deserialize an object from bytes.

        - data: the bytes to deserialize the object from.
        
        See `load` for the meaning of the other arguments.

        """
        return self.load(io.BytesIO(data), header = header)


serializer = Serializer()
dump = serializer.dump
dumps = serializer.dumps
load = serializer.load
loads = serializer.loads

__all__ = [
    'dump', 'dumps', 'load', 'loads',
    'Serializer', 'serializer',
    'writer', 'reader',
]

### Local Variables:
### fill-column: 90
### End:
