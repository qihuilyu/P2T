function [sino, inter, Sino3Dbuff] = rebin_PET_TOF(reconparams, detid_pair, deltat, TR)

nb_cryst = reconparams.nb_cryst;
R1 = reconparams.R1;
distrange = reconparams.distrange;

clight = 300; % c = 300mm/ns
deltar = deltat*clight/2; 

theta = detid_pair/nb_cryst*2*pi;
x_ = R1*sin(theta);
y_ = R1*cos(theta);

p1 = [x_(:,1), y_(:,1)];
p2 = [x_(:,2), y_(:,2)];
p1_p2 = p1 - p2;
pC = (p1+p2)/2;
n_p1_p2 = p1_p2./sqrt(sum(p1_p2.^2,2));
p0 = -n_p1_p2.*deltar + (p1+p2)/2;

%% Show image
p0(sqrt(sum(p0.^2,2))>distrange) = NaN;

maskp0 = isnan(p0(:,1));
pC(maskp0,:) = [];
p0(maskp0,:)=[];
p1_p2(maskp0,:)=[];

dist = round(sqrt(sum(pC.^2,2)),4);
xdy = p1_p2(:,1)./p1_p2(:,2);
ang = mod(round(atan(xdy),4),3.1416);

[uniang, ~, indang] = unique(ang);

xC = pC(:,1);
yC = pC(:,2);
dist(yC<0) = -dist(yC<0);
dist(yC==0) = dist(yC==0).*sign(xC(yC==0));
[unidist, ~, inddist] = unique(dist);

inter = max(abs(diff(unidist)));
maxabsunidist = max(abs(unidist));
newunidist = linspace(-maxabsunidist,maxabsunidist,floor(2*maxabsunidist/inter))';

p0_pC = p0-pC;
xp0pC = p0_pC(:,1);
yp0pC = p0_pC(:,2);
distp0 = round(sqrt(sum((p0_pC).^2,2)),4);
distp0(xp0pC<0) = -distp0(xp0pC<0);
distp0(xp0pC==0) = distp0(xp0pC==0).*sign(yp0pC(xp0pC==0));

nd0 = numel(unidist);
nd = numel(newunidist);
na = numel(uniang);
Sino3Dbuff = zeros(nd0,nd,na);

sigma = clight*TR/2;
gaussFilter = exp(-(newunidist'-distp0) .^ 2 / (2 * sigma ^ 2));
dx = newunidist'-distp0;
gaussFilter(dx<-3*sigma|dx>3*sigma) = 0;
gaussFilter = gaussFilter./sum (gaussFilter,2); % normalize
% figure;plot(gaussFilter(12355,:))

for ii = 1:length(distp0)
    id = inddist(ii);
    ia = indang(ii);
    ikernel = gaussFilter(ii,:);
    if(~any(isnan(ikernel)))
        Sino3Dbuff(id,:,ia) = Sino3Dbuff(id,:,ia) + ikernel;
    end
end

sumbuff1 = sum(sum(Sino3Dbuff(1:2:end,:,1)));
sumbuff2 = sum(sum(Sino3Dbuff(2:2:end,:,1)));

sino = zeros(nd,nd,na);
for ii = 1:na
    for jj = 1:nd
        if(mod(ii,2)==1)
            if(sumbuff2==0 & sumbuff1>0)
                flag=1;
            else
                flag=2;
            end
        else
            if(sumbuff1==0 & sumbuff2>0)
                flag=1;
            else
                flag=2;
            end            
        end
        
        if(flag==1)
            currentsino = Sino3Dbuff(1:2:end,jj,ii);
            currentdist = unidist(1:2:end);
            vq1 = interp1(currentdist,currentsino,newunidist,'linear');
%             figure(100);plot(currentdist,currentsino,'o',newunidist,vq1,':.'); hold on;
        elseif(flag==2)
            currentsino = Sino3Dbuff(2:2:end,jj,ii);
            currentdist = unidist(2:2:end);
            vq1 = interp1(currentdist,currentsino,newunidist,'linear');
%             figure(100);plot(currentdist,currentsino,'o',newunidist,vq1,':.'); hold on;
        end
        sino(:,jj,ii) = vq1;
    end
end
sino(isnan(sino)) = 0;
sino(sino<0) = 0;
% figure;imshow(Sino,[])
