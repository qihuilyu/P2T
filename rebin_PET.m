function [sino, inter, Sinobuff] = rebin_PET(detid_pair, numdet, R1, distrange)

[uniang, indang, unidist, inddist, ~] = list2sino(detid_pair, numdet, R1, distrange);

Sinobuff = full(sparse(inddist,indang,1));
disp(['Number of detected LORs: ' num2str(sum(Sinobuff(:)))])

maxunidist = max(unidist);
minunidist = min(unidist);
inter = (maxunidist - minunidist)/(numel(unidist)-1);
newunidist = (minunidist:inter:maxunidist)';

nd = numel(unidist);
na = numel(uniang);
sino = zeros(nd,na);
for ii = 1:na
    Ind = find(Sinobuff(:,ii)>0);
    %     figure(102); plot(unidist(Ind),Sinobuff(Ind,ii),'o'); hold on;
    %     plot(newunidist,vq1,':.')
    if(numel(Ind)>1)
        vq1 = interp1(unidist(Ind),Sinobuff(Ind,ii),newunidist,'linear');
    elseif(numel(Ind)==1)
        [~,minInd] = min(abs(unidist(Ind)-newunidist));
        vq1 = zeros(nd,1);
        vq1(minInd) = unidist(Ind);
    else
        vq1 = zeros(nd,1);
    end
    sino(:,ii) = vq1;
end
sino(isnan(sino)) = 0;
sino(sino<0) = 0;
% figure;imshow(Sino,[])

