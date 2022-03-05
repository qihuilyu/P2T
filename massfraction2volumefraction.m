function VF = massfraction2volumefraction(density, massfraction)

VF = (massfraction(:)./density(:))/(sum(massfraction(:)./density(:)));




