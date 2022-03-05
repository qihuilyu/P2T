import struct
from abc import abstractclassmethod
import numpy as np
import scipy.sparse as sparse
import h5py

def load_sparse_data(fname):
    with open(fname, 'rb') as fd:
        magic = b''
        for b in struct.unpack('4c', fd.read(4)):
            magic += b
        if magic != b'\x00\x00\xae\xfd':
            raise RuntimeError("Filetype does not match expected for sparse data")

        # load array size
        size = struct.unpack('3I', fd.read(12))

        # load nnz
        nnz = struct.unpack('L', fd.read(8))[0]

        # load indices
        index = struct.unpack('{:d}L'.format(nnz), fd.read(nnz*8))

        # load values
        value = struct.unpack('{:d}d'.format(nnz), fd.read(nnz*8))
    return size, index, value

def sparse2dense(size, index, value):
    """convert sparse data (COO) to dense array"""
    arr = np.zeros(np.product(size))
    for idx, val in zip(index, value):
        if idx>= arr.size: break # hack to fix off-by-one error in simulation code that allowed one extra element to be stored
        arr[idx] = val
    return arr.reshape(size[::-1])


class SparseMatrixBase():
    def __init__(self, outfile, *args, drop_thresh=1e-4, **kwargs):
        self.outfile = outfile
        self.drop_thresh = drop_thresh
        self.coo = None

    @abstractclassmethod
    def add_column(self, vol):
        # invalidate coo cache
        self.coo = None
        pass

    @abstractclassmethod
    def tocoo(self):
        pass

    def tocsc(self):
        return self.tocoo().tocsc()

    def tocsr(self):
        return self.tocoo().tocsr()

class SparseMatrixCOO(SparseMatrixBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = [] # non-zero values
        self.rowi = [] # row indices
        self.coli = [] # col indices
        self.nrows = 0 # y-dimension (M)
        self.ncols = 0 # x-dimension (N)
        self._buf_limit = (1024.0)**3 * 1.0/3.0 # 1GB total bufsize (split between three arrays)
        self._chunksize = (10000,)

        self.initialized = False

    @classmethod
    def fromFile(cls, infile):
        self = cls(infile)
        self._read_from_file()
        return self

    def add_column(self, vol):
        if not self.initialized:
            self._init_h5file()

        super().add_column(vol)

        if vol is None:
            self.ncols += 1
            return

        # flush data after reaching buf size limit
        if len(self.data) >= self._buf_limit:
            self._flush_buf_to_file()

        # flatten vol to column
        nrows = vol.size
        if self.nrows and self.nrows != nrows:
            raise RuntimeError("number of rows in column doesn't match existing columns")
        elif self.nrows == 0:
            # set column size for first column addition
            self.nrows = nrows

        if np.all(vol == 0):
            self.ncols += 1
            return

        data = np.ravel(vol).T

        # filter small values
        if self.drop_thresh is not None:
            tmp = data+np.amin(data)
            tmax = np.amax(tmp)
            if tmax != 0:
                tmp /= tmax
                data[tmp<self.drop_thresh] = 0.0

        # convert to sparse column
        row_indices = data.nonzero()[0].tolist()
        self.rowi.extend( row_indices                   )
        self.coli.extend( [self.ncols]*len(row_indices) )
        self.data.extend( data[row_indices].tolist()        )
        self.ncols += 1

    def tocoo(self):
        if self.coo is not None:
            return self.coo
        if self.initialized:
            # need to write remainder of buffer first
            self.finish()
            self._read_from_file()
        self.coo = sparse.coo_matrix((self.data, (self.rowi, self.coli)), shape=(self.nrows, self.ncols))
        return self.coo

    def _init_h5file(self):
        """opens h5file and inits resizable, chunked datasets for incremental flushes to
        reduce runtime memory usage"""
        self.h5file = h5py.File(self.outfile, 'w')
        self._dsdata  = self.h5file.create_dataset('data',  shape=(0,), maxshape=(None,), chunks=self._chunksize, dtype='f4')
        self._dsrowi  = self.h5file.create_dataset('i',     shape=(0,), maxshape=(None,), chunks=self._chunksize, dtype='u4')
        self._dscoli  = self.h5file.create_dataset('j',     shape=(0,), maxshape=(None,), chunks=self._chunksize, dtype='u4')
        self._dsnrows = self.h5file.create_dataset('nrows', shape=(), dtype='u4')
        self._dsncols = self.h5file.create_dataset('ncols', shape=(), dtype='u4')
        self.initialized = True

    def _read_from_file(self):
        with h5py.File(self.outfile, 'r') as fd:
            self.data = fd['data'][()]
            self.rowi = fd['i'][()]
            self.coli = fd['j'][()]
            self.nrows = fd['nrows'][()]
            self.ncols = fd['ncols'][()]

    def _flush_buf_to_file(self):
        assert len(self.data) == len(self.rowi)
        assert len(self.data) == len(self.coli)
        if len(self.data):
            # resize datasets in h5file
            self._dsdata.resize((self._dsdata.len()+len(self.data), ))
            self._dsrowi.resize((self._dsrowi.len()+len(self.data), ))
            self._dscoli.resize((self._dscoli.len()+len(self.data), ))
            # copy memory buffer to h5file
            self._dsdata[-len(self.data):] = self.data
            self._dsrowi[-len(self.data):] = self.rowi
            self._dscoli[-len(self.data):] = self.coli
            # reset memory buffers
            self.data = []
            self.rowi = []
            self.coli = []

    def _finalize_h5file(self):
        # check if file is open and writable
        if self.h5file:
            self._flush_buf_to_file()
            self._dsnrows[()] = self.nrows
            self._dsncols[()] = self.ncols
            self.h5file.create_dataset('sparse_threshold', shape=(), dtype='f4', data=self.drop_thresh if self.drop_thresh is not None else 0.0)
            self.h5file.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.finish()

    def finish(self):
        self._finalize_h5file()
