import scipy.sparse
import numpy as np

from .roundtrip import *

def test_scipy():
    from safeserialize.scipy import dumps, loads
    np.random.seed(0)
    m = 20
    n = 10
    A = np.random.rand(m, n)
    A[np.random.rand(m, n) < 0.9] = 0

    matrices = {
        "bsr": scipy.sparse.bsr_matrix(A),
        "csr": scipy.sparse.csr_matrix(A),
        "csc": scipy.sparse.csc_matrix(A),
        "coo": scipy.sparse.coo_matrix(A),
    }

    for value in matrices.values():
        roundtrip_sparse(value)
