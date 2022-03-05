function [uniang, indang, unidist, inddist, gooddetind] = list2sino(detid_pair, numdet, R1, distrange)

theta = detid_pair/numdet*2*pi;
x_ = R1*sin(theta);
y_ = R1*cos(theta);

p1 = [x_(:,1), y_(:,1)];
p2 = [x_(:,2), y_(:,2)];
pC = (p1+p2)/2;
dist = round(sqrt(sum(pC.^2,2)),4);

p1_p2 = p1 - p2;
xdy = p1_p2(:,1)./p1_p2(:,2);
ang = mod(atan(xdy),pi);

gooddetind = find((~isnan(ang)) & dist<distrange);
ang = ang(gooddetind);
dist = dist(gooddetind);
pC = pC(gooddetind,:);
uniang = (0:numdet-1)/numdet*pi;
indang = mod(round(ang/(pi/numdet)),numdet)+1;

xC = pC(:,1);
yC = pC(:,2);
dist(yC<0) = -dist(yC<0);
dist(yC==0) = -dist(yC==0).*sign(xC(yC==0));
[unidist, ~, inddist] = unique(dist);

inddist = max(inddist) + 1 - inddist;
unidist = flip(unidist);
