
nx = 153;
ny = 141;
nz = 1;

[x,y,z] = ndgrid(1:nx,1:ny,1:nz);
mask0 = ((x-(1+nx)/2).^2/(60.^2)+(y-(1+ny)/2).^2/(50.^2)<1);

% phantom_tumorwithAuCa = ones(nx,ny,nz)*0;
% mask0 = ((x-(1+nx)/2).^2/(60.^2)+(y-(1+ny)/2).^2/(50.^2)<1);
% phantom_tumorwithAuCa(mask0) = 1;
% figure;imshow3D(phantom_tumorwithAuCa,[]);

slicenum = 50;
x_CT = img(:,:,end+1-slicenum)-1000;
mumap_10MV = 0*x_CT;
mumap_10MV(x_CT>=0) = 0.004942;
mumap_10MV(x_CT<0) = 0;
fluence = 0*mumap_10MV;

numbeams = 20;
deltatheta = 2*pi/numbeams;
iso = [(nx+1)/2*imgres (ny+1)/2*imgres (nz+1)/2*imgres];
nbl = 50; count = 1;
for theta = 0.5*deltatheta:deltatheta:2*pi
    src = 1000*[cos(theta) sin(theta) 0] + iso;
    
    img_fluence = 0*mumap_10MV;
    for ib = -nbl/2*5:5:nbl/2*5
        tariso = ib*[-sin(theta) cos(theta) 0] + iso;
        tar = 2*tariso-src;
        %     tar = [-5,-5,0] + iso;
        
        resolution.x = imgres;
        resolution.y = imgres;
        resolution.z = imgres;
        cubes{1} = mumap_10MV;
        [alphas,l,rho,d12,ix] = matRad_siddonRayTracer([0 0 0], ...
            resolution, ...
            src, ...
            tar, ...
            cubes);
        
        [i,j] = ind2sub([nx,ny],ix);
        imgl = 0*mumap_10MV;
        imgl(ix) = l;
        
        imglmu = imgl.*mumap_10MV;
        
        imgseq = 0*imglmu;
        imgseq(ix) = 1:numel(ix);
        
        attenbuff = exp(-cumsum([0,imglmu(ix)]));
        img_fluence(ix) = attenbuff(1:end-1);
        if mod(count,5)==0
            figure(5);imshow(imglmu,[])
            figure(6);imshow(imgseq,[])
            figure(7);imshow(img_fluence,[])
        end
        count = count + 1;
    end
    
    img_fluence(img_fluence==0) = NaN;
    img_fluence_filled = fillmissing(img_fluence,'nearest');
    figure(8);imshow(img_fluence_filled,[])
    
    fluence = fluence + img_fluence_filled;
    figure(9);imshow(fluence,[])
end

fluence2 = fluence;
fluence2(mask0==0) = 0;
figure(10);imshow(fluence2,[])

fluence2 = imgaussfilt(fluence2);
figure(10);imshow(fluence2,[])
figure(10);imshow(1./fluence2,[0,0.1])


Anni3D = reshape(full(sum(M_Anni,2)),size(masks{1}.mask));
Anni2D = Anni3D(:,:,slicenum);
figure;imshow(Anni2D,[])
Anni2D_corrected = Anni2D./fluence2*50000000;
Anni2D_corrected(mask0==0) = 0;
figure(10);imshow(Anni2D_corrected,[102,137])



