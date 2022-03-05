function mixturedensity = compute_mixturedensity(density, massfraction)

mixturedensity = 1/sum(massfraction(:)./density(:));


