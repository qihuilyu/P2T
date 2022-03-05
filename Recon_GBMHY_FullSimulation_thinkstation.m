clear
close all
clc

patientName = 'GBMHY_final_100m_run01torun10_1000MUpermin';
projectName = 'PairProd';
patFolder = fullfile('D:\datatest\PairProd\',patientName);
OutputFileName = fullfile('D:\datatest\PairProd\','GBMHY.mat');
% CERR('CERRSLICEVIEWER')
% sliceCallBack_QL('OPENNEWPLANC', OutputFileName);

projectFolder = fullfile(patFolder,projectName);
dosecalcFolder = fullfile(patFolder,'dosecalc');
dosematrixFolder = fullfile(projectFolder,'dosematrix');
resultsFolder = fullfile(projectFolder,'results');
mkdir(resultsFolder)

load(fullfile(dosematrixFolder,[patientName projectName '_ringdetection_directmerge.mat']),...
    'energy','detectorIds','CorrectedTime','eventIds','numeventsvec');
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

%% Image Reconstruction
TimeResolution = 0.3; % 300 ps
CorrectedTime_TR = CorrectedTime + TimeResolution*randn(size(CorrectedTime));
Ind_coin_accept = IdentifyLOR(energy, CorrectedTime_TR, CoincidenceTime, EnergyResolution);
TruePositive = length(intersect(Ind_coin_511(:,1).*Ind_coin_511(:,2),Ind_coin_accept(:,1).*Ind_coin_accept(:,2)))/length(Ind_coin_accept(:,1));

R1 = 1200;
distrange = 300;
imgsize = size(img);
nb_cryst = max(detectorIds);

detid_pair = detectorIds(Ind_coin_accept);
[sino, dr, newunidist, sinobuff, unidist] = rebin_PET2(detid_pair, nb_cryst, R1, distrange);

imgres = 2.5;
ig = image_geom('nx', size(img,1), 'ny', size(img,2), 'fov', size(img,1)*imgres);
sg = sino_geom('par', 'nb', size(sino,1), 'na', size(sino,2), 'dr', dr);
img_fbp_nocorrect = em_fbp_QL(sg, ig, sino);
figure;imshow(img_fbp_nocorrect,[])

Anni3D = reshape(M_Anni*numeventsvec,size(masks{1}.mask));
Anni2D = Anni3D(:,:,slicenum);
figure;imshow(Anni2D,[])

dose3D = reshape(M*numeventsvec,size(masks{1}.mask));
dose2D = dose3D(:,:,slicenum);
figure;imshow(dose2D,[])

ind1 = 20; ind2 = 20;
Anni2D = TranslateFigure(Anni3D(:,:,slicenum),ind1,ind2);
% figure;imshow([Anni2D/max(Anni2D(:)),img_fbp_nocorrect/max(img_fbp_nocorrect(:))],[])
C = imfuse(Anni2D/max(Anni2D(:)),img_fbp_nocorrect/max(img_fbp_nocorrect(:)),'falsecolor','Scaling','joint','ColorChannels',[1 2 0]);
figure; imshow(C)

mumapnew = TranslateFigure(mumap,ind1,ind2);
figure;imshow(mumapnew,[])

G = Gtomo2_strip(sg, ig);
% ci = GetACfactor_sino(G, mumap);
li = G * mumapnew;
ci = exp(-li);
img_fbp = em_fbp_QL(sg, ig, sino./ci);
figure;imshow([img_fbp],[])
saveas(gcf,fullfile(resultsFolder,['img_fbp.png']));

save(fullfile(resultsFolder,['Recon_pairprod_fbp.mat']),...
    'Anni2D','img_fbp',...
    'ci','TruePositive');

%% Reconstruction-less image generation
TimeResolution = 0.02; % 20 ps
CorrectedTime_TR = CorrectedTime + TimeResolution*randn(size(CorrectedTime));
Ind_coin_accept = IdentifyLOR(energy, CorrectedTime_TR, CoincidenceTime, EnergyResolution);
TruePositive = length(intersect(Ind_coin_511(:,1).*Ind_coin_511(:,2),Ind_coin_accept(:,1).*Ind_coin_accept(:,2)))/length(Ind_coin_accept(:,1));
detid_pair = detectorIds(Ind_coin_accept);

cilist = GetACfactor_list(ci, detid_pair, nb_cryst, R1, distrange, newunidist);
reconparams = struct('nb_cryst',nb_cryst,'R1',R1,'distrange',distrange,...
    'imgres',imgres,'imgsize',[ig.nx, ig.ny]);
deltat = CorrectedTime(Ind_coin_accept(:,1)) - CorrectedTime(Ind_coin_accept(:,2));
for sigma0 = 2
    [img_direct, img_ci, img_ciN] = recon_TOF_direct(reconparams, detid_pair, deltat, sigma0, cilist);
    figure;imshow([img_direct],[])
    saveas(gcf,fullfile(resultsFolder,['img_direct_sigma0_' num2str(sigma0) '.png']));
    
    save(fullfile(resultsFolder,['Recon_pairprod_direct_sigma0_' num2str(sigma0) '.mat']),...
        'Anni2D','img_direct',...
        'ci','cilist','TruePositive');
end
% figure;imshow([img_fbp/max(img_fbp(:)) img_direct/max(img_direct(:))],[])
% figure;imshow([img_fbp_nocorrect/max(img_fbp_nocorrect(:)) img_direct/max(img_direct(:))],[])

%% TERMA

load('D:\datatest\PairProd\GBMHY_TERMA_100m\PairProd\dosematrix\GBMHY_TERMA_100mPairProd_M_TERMA_HighRes.mat','M_TERMA')
load('D:\datatest\PairProd\GBMHY_final_100m\PairProd\dosematrix\GBMHY_final_100mPairProd_M_HighRes.mat')

TERMA3D = reshape(M_TERMA*numeventsvec,size(masks{1}.mask));
TERMA2D = TERMA3D(:,:,slicenum);
figure;imshow(TERMA2D,[])

Anni3D = reshape(M_Anni*numeventsvec,size(masks{1}.mask));
Anni2D = Anni3D(:,:,slicenum);
figure;imshow(Anni2D,[])

dose3D = reshape(M*numeventsvec,size(masks{1}.mask));
dose2D = dose3D(:,:,slicenum);
figure;imshow(dose2D,[])

save(fullfile(resultsFolder,['Dose_TERMA_PPTI.mat']),...
    'Anni3D','TERMA3D','dose3D');


