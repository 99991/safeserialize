from ..core import writer, reader, write, read

_allowed_dtypes = {
    "bool",
    "uint8", "uint16", "uint32", "uint64",
    "int8", "int16", "int32", "int64",
    "float16", "float32", "float64",
    "complex64", "complex128",
}

VERSION = 1

@writer("torch.Tensor")
def write_tensor(data, out):
    write(VERSION, out)
    device = data.device
    write(str(device), out)
    data_np = data.detach().cpu().numpy()
    assert str(data_np.dtype) in _allowed_dtypes
    write(data_np, out)

@reader("torch.Tensor")
def read_tensor(f):
    version = read(f)
    assert version == VERSION
    device = read(f)
    import torch
    data = read(f)
    assert str(data.dtype) in _allowed_dtypes
    return torch.from_numpy(data).to(device)
