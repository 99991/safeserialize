import pytest
from .roundtrip import *

def test_float():
    from safeserialize.torch import dumps, loads
    import torch
    torch.manual_seed(0) # This stays in effect for the following tests as well.
    x = torch.rand(2, 3, 4)

    roundtrip_torch(x)

def test_cuda():
    from safeserialize.torch import dumps, loads
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.rand(2, 3, 4)

    x = x.to(device)

    roundtrip_torch(x)

def test_long():
    from safeserialize.torch import dumps, loads
    import torch
    x = torch.arange(5)

    roundtrip_torch(x)

def test_half():
    from safeserialize.torch import dumps, loads
    import torch
    x = torch.rand(5).half()

    roundtrip_torch(x)

def test_transposed():
    from safeserialize.torch import dumps, loads
    import torch
    A = torch.rand(2, 3)

    roundtrip_torch(A)
