function [s] = read_sparse_mcdose(h5file)
%READ_SPARSE_MCDOSE load M-matrix from MC generated file
data  = double(h5read(h5file, '/data' )  );
i     = double(h5read(h5file, '/i'    )+1);
j     = double(h5read(h5file, '/j'    )+1);
nrows = double(h5read(h5file, '/nrows')  );
ncols = double(h5read(h5file, '/ncols')  );

s = sparse(i, j, data, nrows, ncols);
end