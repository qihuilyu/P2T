
nx = 153;
ny = 141;
nz = 100;

airHU = -1000;
waterHU = 0;

[x,y,z] = ndgrid(1:nx,1:ny,1:nz);

SquarePhantom = ones(nx,ny,nz)*waterHU;
mask0 = (z<80 & z>20);
SquarePhantom(mask0) = waterHU;
figure;imshow3D(SquarePhantom,[-1000,1600])

BODY = mask0;
figure;imshow3D(BODY,[])

PTV = BODY;

mkdir('D:\datatest\PairProd\SquarePhantom\')
save('D:\datatest\PairProd\SquarePhantom\SquarePhantom.mat','SquarePhantom','BODY','PTV')

