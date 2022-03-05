function [img_direct, img_ci, img_ciN] = recon_TOF_direct(reconparams, detid_pair, deltat, sigma0, varargin)

if(~isempty(varargin))
    cilist = varargin{1};
else
    cilist = ones(size(detid_pair,1),1);
end

nb_cryst = reconparams.nb_cryst;
R1 = reconparams.R1;
distrange = reconparams.distrange;
imgres = reconparams.imgres;
imgsize = reconparams.imgsize;

clight = 300; % c = 300mm/ns
deltar = deltat*clight/2; 

theta = detid_pair/nb_cryst*2*pi;
x_ = R1*sin(theta);
y_ = R1*cos(theta);

p1 = [x_(:,1), y_(:,1)];
p2 = [x_(:,2), y_(:,2)];
p1_p2 = p1 - p2;
n_p1_p2 = p1_p2./sqrt(sum(p1_p2.^2,2));
p0 = -n_p1_p2.*deltar + (p1+p2)/2;

%% Show image
p0(sqrt(sum(p0.^2,2))>distrange) = NaN;
cilist(isnan(p0(:,1))) = [];
p0(isnan(p0(:,1)),:)=[];
xp0 = p0(:,1);
yp0 = p0(:,2);

img_direct = zeros(imgsize(1),imgsize(2));
img_ci = zeros(imgsize(1),imgsize(2));
img_ciN = zeros(imgsize(1),imgsize(2));
xImage = ((1:imgsize(1))-(1+imgsize(1))/2)*imgres;
yImage = ((1:imgsize(2))-(1+imgsize(2))/2)*imgres;

[x,y] = ndgrid(1:imgsize(1),1:imgsize(2));
sigma = sigma0/imgres;

for ii = 1:length(xp0)
    ixp0 = xp0(ii);
    iyp0 = yp0(ii);
    
    if(ixp0<min(xImage)||ixp0>max(xImage)||iyp0<min(yImage)||iyp0>max(yImage))
         continue
    end
    
    [~,xind] = min(abs(ixp0-xImage));
    [~,yind] = min(abs(iyp0-yImage));
    
    if(sigma==0)
        img_direct(xind,yind) = img_direct(xind,yind) + 1/cilist(ii);
    else
        exponent = ((x-xind).^2 + (y-yind).^2)./(2*sigma^2);
        val       = (exp(-exponent));
        img_direct = img_direct + val/cilist(ii);
    end
    
    img_ci(xind,yind) = (img_ci(xind,yind)*img_ciN(xind,yind) + cilist(ii))/(img_ciN(xind,yind) + 1);
    img_ciN(xind,yind) = img_ciN(xind,yind) + 1;

end





