function ImgNew = TranslateFigure(ImgOld,ind1,ind2)

ImgNew = 0*ImgOld;

if(ind1>=0 && ind2>=0)
    ImgNew(ind1+1:end,ind2+1:end) = ImgOld(1:end-ind1,1:end-ind2);
elseif(ind1>=0 && ind2<0)
    ImgNew(ind1+1:end,1:end+ind2) = ImgOld(1:end-ind1,-ind2+1:end);
elseif(ind1<0 && ind2>=0)
    ImgNew(1:end+ind1,ind2+1:end) = ImgOld(-ind1+1:end,1:end-ind2);
elseif(ind1<0 && ind2<0)
    ImgNew(1:end+ind1,1:end+ind2) = ImgOld(-ind1+1:end,-ind2+1:end);
end

