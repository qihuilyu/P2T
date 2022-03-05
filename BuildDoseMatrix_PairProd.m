function [M,M_Anni,dose_data,masks]=BuildDoseMatrix_PairProd(h5file, maskfile, fmapsfile, Anni3Dfile)

verbose = 2;
PTVind = 1;

[ masks ] = open_masks( maskfile, 'xyz', verbose );
PTV = masks{PTVind}.mask;
sz = size(PTV);

for i = 1:length(masks)
    masks{i}.mask = permute(masks{i}.mask,[2,1,3]);
    masks{i}.name = masks{i}.name;
end

M = read_sparse_mcdose(h5file);

dose_data = read_fmaps(fmapsfile, verbose, 'all');

[r,c,v] = find(M);
[Nrows, Ncols] = size(M);
dicomsize = sz;
[I,J,K] = ind2sub(dicomsize,r);
dicomsizenew = dicomsize;
dicomsizenew(2) = dicomsize(1);
dicomsizenew(1) = dicomsize(2);
r = sub2ind(dicomsizenew,J,I,K);

M = sparse(r, c, v, Nrows, Ncols); %OUTVAR


M_Anni = read_sparse_mcdose(Anni3Dfile);

[r,c,v] = find(M_Anni);
[Nrows, Ncols] = size(M_Anni);
dicomsize = sz;
[I,J,K] = ind2sub(dicomsize,r);
dicomsizenew = dicomsize;
dicomsizenew(2) = dicomsize(1);
dicomsizenew(1) = dicomsize(2);
r = sub2ind(dicomsizenew,J,I,K);

M_Anni = sparse(r, c, v, Nrows, Ncols); %OUTVAR


