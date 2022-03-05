function [sino, inter, newunidist, Sinobuff, unidist] = rebin_PET2(detid_pair, numdet, R1, distrange, varargin)

[uniang, indang, unidist, inddist, gooddetind] = list2sino(detid_pair, numdet, R1, distrange);

if(isempty(varargin))
    optimizedweights = 1;
else
    optimizedweights = varargin{1};
    optimizedweights = optimizedweights(gooddetind);
end


Sinobuff = full(sparse(inddist,indang,optimizedweights));
disp(['Number of detected LORs: ' num2str(sum(Sinobuff(:)))])

inter = max(abs(diff(unidist)));
maxabsunidist = max(abs(unidist));
newunidist = flip(linspace(-maxabsunidist,maxabsunidist,floor(2*maxabsunidist/inter)))';

nd = numel(newunidist);
na = numel(uniang);
sino = zeros(nd,na);

sumbuff1 = sum(sum(Sinobuff(1:2:end,1)));
sumbuff2 = sum(sum(Sinobuff(2:2:end,1)));

for ii = 1:na
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
        currentsino = Sinobuff(1:2:end,ii);
        currentdist = unidist(1:2:end);
        vq1 = interp1(currentdist,currentsino,newunidist,'linear');
    elseif(flag==2)
        currentsino = Sinobuff(2:2:end,ii);
        currentdist = unidist(2:2:end);
        vq1 = interp1(currentdist,currentsino,newunidist,'linear');
    end
    sino(:,ii) = vq1;
end
sino(isnan(sino)) = 0;
sino(sino<0) = 0;
% figure;imshow(sino,[])

