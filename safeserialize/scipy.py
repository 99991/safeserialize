from . import *
from . import numpy

VERSION = 1

@writer("scipy.sparse._bsr.bsr_matrix")
def write_bsr_matrix(ser, data, out):
    m, n = data.shape
    ser._write(VERSION, out)
    ser._write(m, out)
    ser._write(n, out)
    ser._write(data.indptr, out)
    ser._write(data.indices, out)
    ser._write(data.data, out)

@reader("scipy.sparse._bsr.bsr_matrix")
def read_bsr_matrix(ser, f):
    version = ser._read(f)
    assert version == VERSION
    m = ser._read(f)
    n = ser._read(f)
    indptr = ser._read(f)
    indices = ser._read(f)
    data = ser._read(f)
    import scipy.sparse
    return scipy.sparse.bsr_matrix((data, indices, indptr), shape=(m, n))

@writer("scipy.sparse._csr.csr_matrix")
def write_csr_matrix(ser, data, out):
    m, n = data.shape
    ser._write(VERSION, out)
    ser._write(m, out)
    ser._write(n, out)
    ser._write(data.indptr, out)
    ser._write(data.indices, out)
    ser._write(data.data, out)

@reader("scipy.sparse._csr.csr_matrix")
def read_csr_matrix(ser, f):
    version = ser._read(f)
    assert version == VERSION
    m = ser._read(f)
    n = ser._read(f)
    indptr = ser._read(f)
    indices = ser._read(f)
    data = ser._read(f)
    import scipy.sparse
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(m, n))

@writer("scipy.sparse._csc.csc_matrix")
def write_csc_matrix(ser, data, out):
    m, n = data.shape
    ser._write(VERSION, out)
    ser._write(m, out)
    ser._write(n, out)
    ser._write(data.indptr, out)
    ser._write(data.indices, out)
    ser._write(data.data, out)

@reader("scipy.sparse._csc.csc_matrix")
def read_csc_matrix(ser, f):
    version = ser._read(f)
    assert version == VERSION
    m = ser._read(f)
    n = ser._read(f)
    indptr = ser._read(f)
    indices = ser._read(f)
    data = ser._read(f)
    import scipy.sparse
    return scipy.sparse.csc_matrix((data, indices, indptr), shape=(m, n))

@writer("scipy.sparse._coo.coo_matrix")
def write_coo_matrix(ser, data, out):
    m, n = data.shape
    ser._write(VERSION, out)
    ser._write(m, out)
    ser._write(n, out)
    row, col = data.coords
    ser._write(row, out)
    ser._write(col, out)
    ser._write(data.data, out)

@reader("scipy.sparse._coo.coo_matrix")
def read_coo_matrix(ser, f):
    version = ser._read(f)
    assert version == VERSION
    m = ser._read(f)
    n = ser._read(f)
    row = ser._read(f)
    col = ser._read(f)
    data = ser._read(f)
    import scipy.sparse
    return scipy.sparse.coo_matrix((data, (row, col)), shape=(m, n))

serializer.harvest(globals())
