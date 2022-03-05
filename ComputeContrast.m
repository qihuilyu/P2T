C = 6;
H = 1;
N = 7;
O = 8;
Na = 11;
P = 30;
S = 16;
Cl = 17;
K = 19;
Mg = 12;
Ca = 20;
Fe = 26;


materials0 = set_default_materialcomp();
materials0.N=0.7;
materials0.O=0.3;
materials0.dens = 1.290e-03;
materials0.name = 'Air';
materials = materials0;

materials0 = set_default_materialcomp();
materials0.H=0.103;
materials0.C=0.105;
materials0.N=0.031;
materials0.O=0.749;
materials0.Na=0.002;
materials0.P=0.002;
materials0.S=0.003;
materials0.Cl=0.002;
materials0.K=0.003;
materials0.dens = 0.217;
materials0.name = 'Lung inhale';
materials(end+1) = materials0;

materials0 = set_default_materialcomp();
materials0.H=0.103;
materials0.C=0.105;
materials0.N=0.031;
materials0.O=0.749;
materials0.Na=0.002;
materials0.P=0.002;
materials0.S=0.003;
materials0.Cl=0.002;
materials0.K=0.003;
materials0.dens = 0.508;
materials0.name = 'Lung exhale';
materials(end+1) = materials0;

materials0 = set_default_materialcomp();
materials0.H=0.114;
materials0.C=0.598;
materials0.N=0.007;
materials0.O=0.278;
materials0.Na=0.001;
materials0.P=0;
materials0.S=0.001;
materials0.Cl=0.001;
materials0.K=0;
materials0.dens = 0.967;
materials0.name = 'Adipose Tissue';
materials(end+1) = materials0;

materials0 = set_default_materialcomp();
materials0.H=0.109;
materials0.C=0.506;
materials0.N=0.023;
materials0.O=0.358;
materials0.Na=0.001;
materials0.P=0.001;
materials0.S=0.001;
materials0.Cl=0.001;
materials0.K=0;
materials0.dens = 0.990;
materials0.name = 'Breast';
materials(end+1) = materials0;

materials0 = set_default_materialcomp();
materials0.H=0.112;
materials0.O=0.888;
materials0.dens = 1;
materials0.name = 'Water';
materials(end+1) = materials0;

materials0 = set_default_materialcomp();
materials0.H=0.102;
materials0.C=0.143;
materials0.N=0.034;
materials0.O=0.710;
materials0.Na=0.001;
materials0.P=0.002;
materials0.S=0.003;
materials0.Cl=0.001;
materials0.K=0.004;
materials0.dens = 1.061;
materials0.name = 'Muscle';
materials(end+1) = materials0;

materials0 = set_default_materialcomp();
materials0.H=0.102;
materials0.C=0.139;
materials0.N=0.030;
materials0.O=0.716;
materials0.Na=0.002;
materials0.P=0.003;
materials0.S=0.003;
materials0.Cl=0.002;
materials0.K=0.003;
materials0.dens = 1.071;
materials0.name = 'Liver';
materials(end+1) = materials0;

materials0 = set_default_materialcomp();
materials0.H=0.085;
materials0.C=0.404;
materials0.N=0.058;
materials0.O=0.367;
materials0.Na=0.001;
materials0.P=0.034;
materials0.S=0.002;
materials0.Cl=0.002;
materials0.K=0.001;
materials0.Mg=0.001;
materials0.Ca=0.044;
materials0.Fe=0.001;
materials0.dens = 1.159;
materials0.name = 'Trabecular Bone';
materials(end+1) = materials0;

materials0 = set_default_materialcomp();
materials0.H=0.056;
materials0.C=0.235;
materials0.N=0.050;
materials0.O=0.434;
materials0.Na=0.001;
materials0.Mg=0.001;
materials0.P=0.072;
materials0.S=0.003;
materials0.Cl=0.001;
materials0.K=0.001;
materials0.Ca=0.146;
materials0.Fe=0;
materials0.dens = 1.575;
materials0.name = 'Dense Bone';
materials(end+1) = materials0;


for ii = 1:numel(materials)
    materials(ii).Z = materials(ii).H*H + materials(ii).C*C + materials(ii).N*N + materials(ii).O*O + materials(ii).Na*Na + materials(ii).P*P + materials(ii).S*S + materials(ii).Cl*Cl + materials(ii).K*K + materials(ii).Mg*Mg + materials(ii).Ca*Ca + materials(ii).Fe*Fe;
    materials(ii).Zcubic = materials(ii).H*H^3 + materials(ii).C*C^3 + materials(ii).N*N^3 + materials(ii).O*O^3 + materials(ii).Na*Na^3 + materials(ii).P*P^3 + materials(ii).S*S^3 + materials(ii).Cl*Cl^3 + materials(ii).K*K^3 + materials(ii).Mg*Mg + materials(ii).Ca*Ca + materials(ii).Fe*Fe;
    materials(ii).rho_Z = materials(ii).Z*materials(ii).dens;
    materials(ii).rho_Zcubic = materials(ii).Zcubic*materials(ii).dens;
end 

waterind = 6;
for ii = 1:numel(materials)
    materials(ii).rho_Z_contrast = (materials(ii).rho_Z/materials(waterind).rho_Z-1)*100;
    materials(ii).rho_Zcubic_contrast = (materials(ii).rho_Zcubic/materials(waterind).rho_Zcubic-1)*100;
end 

% Air*Air_dens/Water-1
% lunginhale*lunginhale_dens/Water-1
% lungexhale*lungexhale_dens/Water-1
% Adipose*Adipose_dens/Water-1
% Breast*Breast_dens/Water-1
% muscle*muscle_dens/Water-1
% liver*liver_dens/Water-1
% trabecularBone*trabecularBone_dens/Water-1
% denseBone*denseBone_dens/Water-1
% materials0.name = 'Lung inhale';
% materials(end+1) = materials0;


T = struct2table(materials);
filename = 'D:\datatest\PairProd\GoodResult\materials.csv';
delete(filename)
writetable(T,filename)






