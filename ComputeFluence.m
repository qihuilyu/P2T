function img_fluence= ComputeFluence(mumap_10MV, src, tardir, iso, imgres)

resolution.x = imgres;
resolution.y = imgres;
resolution.z = imgres;
FOV = 150;

count = 1;
img_fluence = 0*mumap_10MV;
for ib = -FOV:FOV
    tariso = ib*tardir + iso;
    tar = 2*tariso-src;
    
    src2 =  [src(2) src(1) src(3)];
    tar2 =  [tar(2) tar(1) tar(3)];
    
    cubes{1} = mumap_10MV;
    [alphas,l,rho,d12,ix] = matRad_siddonRayTracer([0 0 0], resolution, src2, tar2, cubes);
    imgl = 0*mumap_10MV;
    imgl(ix) = l;
    imglmu = imgl.*mumap_10MV;
        
    attenbuff = exp(-cumsum([0,imglmu(ix)]));
    img_fluence(ix) = attenbuff(1:end-1)./(alphas(1:end-1)*2).^2;
    if mod(count,5)==0
        figure(5);imshow(imglmu,[])
        figure(7);imshow(img_fluence,[])
    end
    count = count + 1;
end


