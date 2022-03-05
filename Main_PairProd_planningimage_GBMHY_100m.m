clear
close all
clc

patientName = 'GBMHY_100m';
projectName = 'PairProd';
patFolder = fullfile('D:\datatest\PairProd\',patientName);
projectFolder = fullfile(patFolder,projectName);
dosecalcFolder = fullfile(patFolder,'dosecalc');
dosematrixFolder = fullfile(projectFolder,'dosematrix');
resultFolder = fullfile(projectFolder,'result');
mkdir(resultFolder)

load(fullfile(dosematrixFolder,[patientName projectName '_ringdetection.mat']),'detectorIds','beamNo','beamletNo','energy','eventIds','globalTimes','CorrectedTime','sortedtime','sortInd');
load(fullfile(dosematrixFolder,[patientName projectName '_M_HighRes.mat']),'M','M_Anni','dose_data','masks');
load(fullfile(dosematrixFolder,[patientName projectName '_dicomimg.mat']),'img','imgres');

%%
load('D:\datatest\PairProd\GBMHY_100m\PairProd\results\GBMHY_100m PairProd Info7 eta0 result.mat')
xPolish = result.xPolish;
StructureInfo = result.StructureInfo;
params = result.params;

slicenum = 88;
x_CT = img(:,:,end+1-slicenum)-1000;
[mumap,densmap,Ind] = lookup_materials_bulk_density(x_CT);

% figure;imshow(mumap,[])
% figure;imshow(densmap,[])
% figure;imshow(Ind,[])
% figure;imshow(x_CT,[])

Anni3D = reshape(full(M_Anni*xPolish),size(StructureInfo(1).Mask));
Anni2D = Anni3D(:,:,slicenum);

dose3D = reshape(full(M*xPolish),size(StructureInfo(1).Mask));
dose2D = dose3D(:,:,slicenum);

ind1 = 20; ind2 = 20;
Anni2Dold = Anni2D;
Anni2D = 0*Anni2D;
Anni2D(ind1+1:end,ind2+1:end) = Anni2Dold(1:end-ind1,1:end-ind2 );

dose2Dold = dose2D;
dose2D = 0*dose2D;
dose2D(ind1+1:end,ind2+1:end) = dose2Dold(1:end-ind1,1:end-ind2 );

mumapold = mumap;
mumap = 0*mumap;
mumap(ind1+1:end,ind2+1:end) = mumapold(1:end-ind1,1:end-ind2 );
% mumap(1:end-ind1,ind2+1:end) = mumapold(ind1+1:end,1:end-ind2 );
figure;imshow(mumap,[])

%% Identify LOR
EnergyResolution = 0.1;
CoincidenceTime = 1;  % ns 

Ind_coin_511 = IdentifyLOR_511(energy, CorrectedTime, CoincidenceTime);
Ind_coin_accept = IdentifyLOR(energy, CorrectedTime, CoincidenceTime, EnergyResolution);

TruePositive = length(Ind_coin_511)/length(Ind_coin_accept);
% save(fullfile(dosematrixFolder,[patientName projectName '_detid_pair.mat']),'Ind_coin_511','Ind_coin_accept');

%%
beamSizes = squeeze(sum(sum(params.BeamletLog0,1),2));
cumsumbeamSizes = cumsum([0; beamSizes]);
beamNoshift = cumsumbeamSizes(beamNo);
beamletIDs = double(beamletNo) + beamNoshift;


%% Image Reconstruction
TimeResolution = 0.4; % 400 ps
CorrectedTime_TR = CorrectedTime + TimeResolution*randn(size(CorrectedTime));
Ind_coin_accept = IdentifyLOR(energy, CorrectedTime_TR, CoincidenceTime, EnergyResolution);

R1 = 1200;
distrange = 300;
imgsize = size(img);
nb_cryst = max(detectorIds);

detid_pair = detectorIds(Ind_coin_accept);
beamletID_select = beamletIDs(Ind_coin_accept(:,1));
optimizedweights = xPolish(beamletID_select);
[sino, dr, newunidist, sinobuff, unidist] = rebin_PET2(detid_pair, nb_cryst, R1, distrange, optimizedweights);

ig = image_geom('nx', size(img,1), 'ny', size(img,2), 'fov', size(img,1)*imgres);
sg = sino_geom('par', 'nb', size(sino,1), 'na', size(sino,2), 'dr', dr);
G = Gtomo2_strip(sg, ig);
% ci = GetACfactor_sino(G, mumap);
li = G * mumap;
ci = exp(-li);
img_fbp = em_fbp_QL(sg, ig, sino./ci);
img_fbp_nocorrect = em_fbp_QL(sg, ig, sino);

%% Reconstruction-less image generation
TimeResolution = 0.02; % 20 ps
CorrectedTime_TR = CorrectedTime + TimeResolution*randn(size(CorrectedTime));
Ind_coin_accept = IdentifyLOR(energy, CorrectedTime_TR, CoincidenceTime, EnergyResolution);

detid_pair = detectorIds(Ind_coin_accept);
beamletID_select = beamletIDs(Ind_coin_accept(:,1));
optimizedweights = xPolish(beamletID_select);
cilist = GetACfactor_list(ci, detid_pair, nb_cryst, R1, distrange, newunidist);
reconparams = struct('nb_cryst',nb_cryst,'R1',R1,'distrange',distrange,...
    'imgres',imgres,'imgsize',[ig.nx, ig.ny]);
deltat = CorrectedTime_TR(Ind_coin_accept(:,1)) - CorrectedTime_TR(Ind_coin_accept(:,2));
[img_direct, img_ci, img_ciN] = recon_TOF_direct(reconparams, detid_pair, deltat, cilist./optimizedweights);
figure;imshow([Anni2D/max(Anni2D(:)) img_fbp_nocorrect/max(img_fbp_nocorrect(:)) img_fbp/max(img_fbp(:)) img_direct/max(img_direct(:))],[])

%%
xoff = -5.2;
yoff = 5.5;
img_fbp_nocorrect(img_fbp_nocorrect<0) = 0;
img_fbp_nocorrect = img_fbp_nocorrect/max(img_fbp_nocorrect(:));
planName = 'img_fbp_nocorrect';
addDoseToGui_Move_QL(repmat(img_fbp_nocorrect,[1,1,size(Anni3D,3)]),[planName],xoff,yoff)

img_fbp(img_fbp<0) = 0;
img_fbp = img_fbp/max(img_fbp(:));
planName = 'img_fbp';
addDoseToGui_Move_QL(repmat(img_fbp,[1,1,size(Anni3D,3)]),[planName],xoff,yoff)

planName = 'img_direct';
addDoseToGui_Move_QL(repmat(img_direct,[1,1,size(Anni3D,3)]),[planName],xoff,yoff)

xoff = -0.2;
yoff = 0.2;
planName = 'Anni3D';
addDoseToGui_Move_QL(Anni3D,[planName],xoff,yoff)

planName = 'dose3D';
addDoseToGui_Move_QL(dose3D,[planName],xoff,yoff)

