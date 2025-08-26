from safeserialize import write, read
from ..core import writer, reader

VERSION = 1

@writer("scipy.sparse._bsr.bsr_matrix")
def write_bsr_matrix(data, out):
    m, n = data.shape
    write(VERSION, out)
    write(m, out)
    write(n, out)
    write(data.indptr, out)
    write(data.indices, out)
    write(data.data, out)

@reader("scipy.sparse._bsr.bsr_matrix")
def read_bsr_matrix(f):
    version = read(f)
    assert version == VERSION
    m = read(f)
    n = read(f)
    indptr = read(f)
    indices = read(f)
    data = read(f)
    import scipy.sparse
    return scipy.sparse.bsr_matrix((data, indices, indptr), shape=(m, n))

@writer("scipy.sparse._csr.csr_matrix")
def write_csr_matrix(data, out):
    m, n = data.shape
    write(VERSION, out)
    write(m, out)
    write(n, out)
    write(data.indptr, out)
    write(data.indices, out)
    write(data.data, out)

@reader("scipy.sparse._csr.csr_matrix")
def read_csr_matrix(f):
    version = read(f)
    assert version == VERSION
    m = read(f)
    n = read(f)
    indptr = read(f)
    indices = read(f)
    data = read(f)
    import scipy.sparse
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(m, n))

@writer("scipy.sparse._csc.csc_matrix")
def write_csc_matrix(data, out):
    m, n = data.shape
    write(VERSION, out)
    write(m, out)
    write(n, out)
    write(data.indptr, out)
    write(data.indices, out)
    write(data.data, out)

@reader("scipy.sparse._csc.csc_matrix")
def read_csc_matrix(f):
    version = read(f)
    assert version == VERSION
    m = read(f)
    n = read(f)
    indptr = read(f)
    indices = read(f)
    data = read(f)
    import scipy.sparse
    return scipy.sparse.csc_matrix((data, indices, indptr), shape=(m, n))

@writer("scipy.sparse._coo.coo_matrix")
def write_coo_matrix(data, out):
    m, n = data.shape
    write(VERSION, out)
    write(m, out)
    write(n, out)
    row, col = data.coords
    write(row, out)
    write(col, out)
    write(data.data, out)

@reader("scipy.sparse._coo.coo_matrix")
def read_coo_matrix(f):
    version = read(f)
    assert version == VERSION
    m = read(f)
    n = read(f)
    row = read(f)
    col = read(f)
    data = read(f)
    import scipy.sparse
    return scipy.sparse.coo_matrix((data, (row, col)), shape=(m, n))
