function [cilist,gooddetind] = GetACfactor_list(cinew, detid_pair, numdet, R1, distrange, newunidist)

[~, indang, unidist, inddist, gooddetind] = list2sino(detid_pair, numdet, R1, distrange);

ndnew = numel(newunidist);
nd = max(inddist);
na = max(indang);
sznew = [ndnew, na];
if(any(sznew-size(cinew)))
    error('Dimension does not match!')
end

sz = [nd, na];
ci = zeros(nd,na);
for ii = 1:na
    ci(:,ii) = interp1(newunidist,cinew(:,ii),unidist,'linear','extrap');
end

Ind = sub2ind(sz,inddist,indang);
cilist = NaN(size(detid_pair,1),1);
cilist(gooddetind) = ci(Ind);
