import torch
from safeserialize import dumps, loads

def test_float():
    x = torch.rand(2, 3, 4)

    roundtrip(x)

def test_cuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.rand(2, 3, 4)

    x = x.to(device)

    roundtrip(x)

def test_long():
    x = torch.arange(5)

    roundtrip(x)

def test_half():
    x = torch.rand(5).half()

    roundtrip(x)

def roundtrip(x):
    s = dumps(x)
    y = loads(s)
    assert torch.equal(x, y)

def test_transposed():
    A = torch.rand(2, 3)

    roundtrip(A)
