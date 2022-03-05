clear
close all
clc

patientName = 'GBMHY_100m';
projectName = 'PairProd';
patFolder = fullfile('D:\datatest\PairProd\',patientName);
projectFolder = fullfile(patFolder,projectName);
dosecalcFolder = fullfile(patFolder,'dosecalc');
dosematrixFolder = fullfile(projectFolder,'dosematrix');
resultFolder = fullfile(projectFolder,'results');
mkdir(resultFolder)

load(fullfile(dosematrixFolder,[patientName projectName '_ringdetection.mat']),'energy','detectorIds','CorrectedTime','eventIds');
load(fullfile(dosematrixFolder,[patientName projectName '_M_HighRes.mat']),'M','M_Anni','dose_data','masks');
load(fullfile(dosematrixFolder,[patientName projectName '_dicomimg.mat']),'img','imgres');

paramsFolder = fullfile(projectFolder,'params');
ParamsNum = 0;
load(fullfile(paramsFolder,['params' num2str(ParamsNum) '.mat']),'params');
InfoNum = 0;
load(fullfile(paramsFolder,['StructureInfo' num2str(InfoNum) '.mat']),'StructureInfo');

slicenum = 88;
x_CT = img(:,:,end+1-slicenum)-1000;
[mumap,densmap,Ind] = lookup_materials_bulk_density(x_CT);

% figure;imshow(mumap,[])
% figure;imshow(densmap,[])
% figure;imshow(Ind,[])
% figure;imshow(x_CT,[])

%% Identify LOR
EnergyResolution = 0.1;
CoincidenceTime = 1;  % ns 

Ind_coin_511 = IdentifyLOR_511(energy, CorrectedTime, CoincidenceTime);
Ind_coin_accept = IdentifyLOR(energy, CorrectedTime, CoincidenceTime, EnergyResolution);

TruePositive = length(Ind_coin_511)/length(Ind_coin_accept);
% save(fullfile(dosematrixFolder,[patientName projectName '_detid_pair.mat']),'Ind_coin_511','Ind_coin_accept');

%% Image Reconstruction
R1 = 1200;
distrange = 300;
imgsize = size(img);
nb_cryst = max(detectorIds);

detid_pair = detectorIds(Ind_coin_accept);
[sino, dr, newunidist, sinobuff, unidist] = rebin_PET2(detid_pair, nb_cryst, R1, distrange);

ig = image_geom('nx', size(img,1), 'ny', size(img,2), 'fov', size(img,1)*imgres);
sg = sino_geom('par', 'nb', size(sino,1), 'na', size(sino,2), 'dr', dr);
img_fbp_nocorrect = em_fbp_QL(sg, ig, sino);

Anni3D = reshape(full(sum(M_Anni,2)),size(masks{1}.mask));
Anni2D = Anni3D(:,:,slicenum);
figure;imshow(Anni2D,[])

dose3D = reshape(full(sum(M,2)),size(masks{1}.mask));
dose2D = dose3D(:,:,slicenum);
figure;imshow(dose2D,[])

ind1 = 20; ind2 = 20;
Anni2D = Anni3D(:,:,slicenum);
Anni2Dold = Anni2D;
Anni2D = 0*Anni2D;
Anni2D(ind1+1:end,ind2+1:end) = Anni2Dold(1:end-ind1,1:end-ind2 );
% Anni2D(1:end+ind1,ind2+1:end) = Anni2Dold(-ind1+1:end,1:end-ind2 );
% figure;imshow([Anni2D/max(Anni2D(:)),img_fbp_nocorrect/max(img_fbp_nocorrect(:))],[])
C = imfuse(Anni2D/max(Anni2D(:)),img_fbp_nocorrect/max(img_fbp_nocorrect(:)),'falsecolor','Scaling','joint','ColorChannels',[1 2 0]);
figure; imshow(C)

mumapold = mumap;
mumap = 0*mumap;
mumap(ind1+1:end,ind2+1:end) = mumapold(1:end-ind1,1:end-ind2 );
% mumap(1:end-ind1,ind2+1:end) = mumapold(ind1+1:end,1:end-ind2 );
figure;imshow(mumap,[])

G = Gtomo2_strip(sg, ig);
% ci = GetACfactor_sino(G, mumap);
li = G * mumap;
ci = exp(-li);
img_fbp = em_fbp_QL(sg, ig, sino./ci);

%% Reconstruction-less image generation
cilist = GetACfactor_list(ci, detid_pair, nb_cryst, R1, distrange, newunidist);
reconparams = struct('nb_cryst',nb_cryst,'R1',R1,'distrange',distrange,...
    'imgres',imgres,'imgsize',[ig.nx, ig.ny]);
deltat = CorrectedTime(Ind_coin_accept(:,1)) - CorrectedTime(Ind_coin_accept(:,2));
[img_direct, img_ci, img_ciN] = recon_TOF_direct(reconparams, detid_pair, deltat, cilist);
figure;imshow([Anni2D/max(Anni2D(:)) img_fbp/max(img_fbp(:)) img_direct/max(img_direct(:))],[])

