function [mumap,densmap,Ind,mumap_3MeV] = lookup_materials_bulk_density(ctnums)
% % lut_dens2mat = [
% %     (0.0,   mat_map["air"]      ),
% %     (0.207, mat_map["lung_in"]  ),
% %     (0.481, mat_map["lung_ex"]  ),
% %     (0.919, mat_map["adipose"]  ),
% %     (0.979, mat_map["breast"]   ),
% %     (1.004, mat_map["water"]    ),
% %     (1.109, mat_map["muscle"]   ),
% %     (1.113, mat_map["liver"]    ),
% %     (1.496, mat_map["bone_trab"]),
% %     (1.654, mat_map["bone_comp"]),
% %     (6.0, mat_map["Tumor"]),
% %     (6.1, mat_map["TumorWithAu005"]),
% %     (6.2, mat_map["TumorWithAu02"]),
% %     (6.3, mat_map["TumorWithAu05"]),
% %     (6.4, mat_map["TumorWithCa005"]),
% %     (6.5, mat_map["TumorWithCa02"]),
% %     (6.5, mat_map["TumorWithCa05"]),
% % ]

inthu = [-5000.0, -1000.0, -400, -150, 100, 300, 2000, 4927, 66000];
intdens = [0.0, 0.01, 0.602, 0.924, 1.075, 1.145, 1.856, 3.379, 7.8];

% huq = -5000:66000;
% densities = interp1(inthu,intdens,huq);
% figure;plot(huq,densities)

densities = interp1(inthu,intdens,ctnums(:));
densmap = reshape(densities,size(ctnums));
matdenslist = [0.0,0.207,0.481,0.919,0.979,1.004,1.109,1.113,1.496,1.654,6.0,6.1,6.2,6.3,6.4,6.5,6.6];

diff = densities - matdenslist;
Ind = reshape(sum(diff>=0,2),size(ctnums));
Ind(Ind>numel(matdenslist)) = numel(matdenslist);
Ind(Ind<1) = 1;

% figure;imshow(reshape(densities,size(ctnums)),[])
% figure;imshow(Ind,[])


%                 air   lung_in   lung_ex  adipose   breast    water     muscle   liver   bone_trab   bone_comp   tumor  TumorAu005  TumorAu02 TumorAu05  TumorCa005  TumorCa02 TumorCa05
maclist =      [0.0086  0.0095   0.0095   0.0096    0.0095    0.0096    0.0095   0.0095     0.0089     0.0089    0.0096    0.0096    0.0096    0.0096      0.0096    0.0096    0.0096];
rholist =      [0.0,    0.207,    0.481,   0.919,    0.979,    1.004,    1.109,   1.113,    1.496,     1.654,    1.004,    1.004,    1.004,    1.004,      1.004,    1.004,    1.004 ];
maclist_3MeV = [0.00397  0.00394  0.00394   0.0119   0.00394   0.00397  0.00393   0.00393   0.00375   0.00375    0.00397   0.00397   0.00397   0.00397    0.00397    0.00397   0.00397];
% mathu = [-1024,   -800,     -522,    -150,     -58,     0,         198,      209,     1140,      1518,    41135,    42516,    43898,    45279,      46660,    48042,    49423];
mulist = maclist.*rholist;
mumap = reshape(mulist(Ind),size(ctnums));

mulist_3MeV = maclist_3MeV.*rholist;
mumap_3MeV = reshape(mulist_3MeV(Ind),size(ctnums));


