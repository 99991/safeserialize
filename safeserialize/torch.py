from . import *
from . import numpy

_allowed_dtypes = {
    "bool",
    "uint8", "uint16", "uint32", "uint64",
    "int8", "int16", "int32", "int64",
    "float16", "float32", "float64",
    "complex64", "complex128",
}

VERSION = 1

@writer("torch.Tensor")
def write_tensor(ser, data, out):
    ser._write(VERSION, out)
    device = data.device
    ser._write(str(device), out)
    data_np = data.detach().cpu().numpy()
    assert str(data_np.dtype) in _allowed_dtypes
    ser._write(data_np, out)

@reader("torch.Tensor")
def read_tensor(ser, f):
    version = ser._read(f)
    assert version == VERSION
    device = ser._read(f)
    import torch
    data = ser._read(f)
    assert str(data.dtype) in _allowed_dtypes
    return torch.from_numpy(data).to(device)

serializer.harvest(globals())
