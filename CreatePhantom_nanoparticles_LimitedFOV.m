
nx = 153;
ny = 141;
nz = 100;

airHU = -1000;
waterHU = 0;

[x,y,z] = ndgrid(1:nx,1:ny,1:nz);

phantom_nanoparticles_LimitedROI = ones(nx,ny,nz)*airHU;
mask0 = ((x-(1+nx)/2).^2/(60.^2)+(y-(1+ny)/2).^2/(50.^2)<1 & z<80 & z>20);
phantom_nanoparticles_LimitedROI(mask0) = waterHU;
figure;imshow3D(phantom_nanoparticles_LimitedROI,[-1000,1600])

inthu = [-5000.0, -1000.0, -400, -150, 100, 300, 2000, 4927, 66000];
intdens = [0.0, 0.01, 0.602, 0.924, 1.075, 1.145, 1.856, 3.379, 7.8];
huq = -5000:66000;
densq = interp1(inthu,intdens,huq);
figure;plot(huq,densq)

matdens = [0.0,0.207,0.481,0.919,0.979,1.004,1.109,1.113,1.496,1.654];
mathu   = [-1024,0,1518,41135,42516,43898,45279,46660,48042,49423];
% matdenslist = [0.0,0.207,0.481,0.919,0.979,1.004,1.109,1.113,1.496,1.654,6.0,6.1,6.2,6.3,6.4,6.5,6.6];
% matdenslist = [6.0,6.1,6.2,6.3,6.4,6.5,6.6];
% for ii = 1:numel(matdenslist)
%     [~,ind] = min((matdenslist(ii)-densq).^2);
%     mathulist(ii) = huq(ind);
% end

r = 35; ninserts = numel(matdens); ri = 5;
for ii = 1:ninserts
    theta = ii/ninserts*2*pi;
    rx = (1+nx)/2 + r*cos(theta);
    ry = (1+ny)/2 + r*sin(theta);
    mask = ((x-rx).^2+(y-ry).^2 <ri.^2 & mask0==1);
    phantom_nanoparticles_LimitedROI(mask) = mathu(ii);
end

figure;imshow3D(phantom_nanoparticles_LimitedROI,[-1000,1600])

BODY = mask0;
figure;imshow3D(BODY,[])

PTV = 0*mask0;
PTV(phantom_nanoparticles_LimitedROI>45279) = 1;
figure;imshow3D(PTV,[])

mkdir('D:\datatest\PairProd\phantom_nanoparticles_LimitedROI\')
save('D:\datatest\PairProd\phantom_nanoparticles_LimitedROI\phantom_nanoparticles_LimitedROI.mat','phantom_nanoparticles_LimitedROI','BODY','PTV')








