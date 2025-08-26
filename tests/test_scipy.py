from safeserialize import dumps, loads

import scipy.sparse
import numpy as np

def test_scipy():
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

    data = dumps(matrices)

    loaded_matrices = loads(data)

    for name, expected_value in matrices.items():
        value = loaded_matrices[name]

        assert (value != expected_value).nnz == 0
