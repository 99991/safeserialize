import torch
torch.manual_seed(0)
from safeserialize import dumps, loads

from .roundtrip import *

def test_float():
    x = torch.rand(2, 3, 4)

    roundtrip_torch(x)

def test_cuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.rand(2, 3, 4)

    x = x.to(device)

    roundtrip_torch(x)

def test_long():
    x = torch.arange(5)

    roundtrip_torch(x)

def test_half():
    x = torch.rand(5).half()

    roundtrip_torch(x)

def test_transposed():
    A = torch.rand(2, 3)

    roundtrip_torch(A)
