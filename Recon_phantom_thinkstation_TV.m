clear
close all
clc

patientName = 'CTphantom_20beam_2mmbeamlet_25m';
beamletwidth = 2;
numevents = 25e+06;
slicenum = 50;

projectName = 'PairProd';
patFolder = fullfile('D:\datatest\PairProd\',patientName);
projectFolder = fullfile(patFolder,projectName);
dosecalcFolder = fullfile(patFolder,'dosecalc');
dosematrixFolder = fullfile(projectFolder,'dosematrix');
resultsFolder = fullfile(projectFolder,'results');
mkdir(resultsFolder)

load(fullfile(dosematrixFolder,[patientName projectName '_ringdetection.mat']),...
    'detectorIds','beamNo','beamletNo','energy','eventIds','globalTimes');
load(fullfile(dosematrixFolder,[patientName projectName '_M_HighRes.mat']),'M','M_Anni','dose_data','masks');
load(fullfile(dosematrixFolder,[patientName projectName '_dicomimg.mat']),'img','imgres');

paramsFolder = fullfile(projectFolder,'params');
ParamsNum = 0;
load(fullfile(paramsFolder,['params' num2str(ParamsNum) '.mat']),'params');
InfoNum = 0;
load(fullfile(paramsFolder,['StructureInfo' num2str(InfoNum) '.mat']),'StructureInfo');

%% Basic information
if(numevents~=max(eventIds) + 99 - mod(max(eventIds)-1,100))
    error('numevents error!!');
end

x_CT = img(:,:,end+1-slicenum)-1000;
[mumap,densmap,Ind] = lookup_materials_bulk_density(x_CT);

% figure;imshow(mumap,[])
% figure;imshow(densmap,[])
% figure;imshow(Ind,[])
% figure;imshow(x_CT,[])

Anni3D = reshape(full(sum(M_Anni,2)),size(masks{1}.mask));
Anni2D = Anni3D(:,:,slicenum);
figure;imshow(Anni2D,[])

dose3D = reshape(full(sum(M,2)),size(masks{1}.mask));
dose2D = dose3D(:,:,slicenum);
figure;imshow(dose2D,[])

mumap_10MV = 0*x_CT;
mumap_10MV(x_CT>=0) = 0.004942;
mumap_10MV(x_CT<0) = 0;

PTV = StructureInfo(1).Mask;
BODY = StructureInfo(2).Mask ==1 | StructureInfo(1).Mask ==1;
mask0 = BODY(:,:,slicenum);
figure;imshow(mask0,[])

BeamletLog0 = params.BeamletLog0;
BeamletInd = BeamletLog0;
numbeamlets = nnz(BeamletLog0);
BeamletInd(BeamletLog0==1) = 1:numbeamlets;
numbeams = size(BeamletLog0,3);
[nx,ny,~] = size(img);
[CenterOfMass] = GetPTV_COM(PTV);
iso = CenterOfMass*imgres;

beamangles = zeros(numbeams,1);
for BeamNo = 1:numbeams
    beamangles(BeamNo) = dose_data.beam_metadata(BeamNo).beam_specs.gantry_rot_rad;
end

%% Compute fluence from each beam
if(exist(fullfile(resultsFolder,'img_fluence_filled.mat'),'file'))
    load(fullfile(resultsFolder,'img_fluence_filled.mat'),'img_fluence_filled');
else
    img_fluence_filled = zeros(nx,ny,numbeams);
    for BeamNo = 1:numbeams
        theta = mod(-beamangles(BeamNo) + 3.1436,2*pi);
        src = 1000*[cos(theta) sin(theta) 0] + iso;
        tardir = [-sin(theta) cos(theta) 0];
        
        img_fluence= ComputeFluence(mumap_10MV, src, tardir, iso, imgres);
        img_fluence(img_fluence==0) = NaN;
        img_fluence_filled(:,:,BeamNo) = inpaint_nans(img_fluence);
        figure(8);imshow(img_fluence_filled(:,:,BeamNo),[])
    end
    save(fullfile(resultsFolder,'img_fluence_filled.mat'),'img_fluence_filled');
end

%% Compute total fluence
fluence = 0*mumap_10MV;
FOV = size(BeamletLog0,1)*beamletwidth/imgres-1;
beamlist = 1:numbeams;
for BeamNo = beamlist
    theta = mod(-beamangles(BeamNo) + 3.1436,2*pi);
    src = 1000*[cos(theta) sin(theta) 0] + iso;
    
    Ind = BeamletInd(:,:,BeamNo);
    Ind = Ind(Ind>0);
    
    xf = zeros(numbeamlets,1);
    xf(Ind) = numevents;
    dose1beam3D = reshape(full(M*xf),size(masks{1}.mask));
    dose1beam = dose1beam3D(:,:,slicenum);
    Anni1beam3D = reshape(full(M_Anni*xf),size(masks{1}.mask));
    Anni1beam = Anni1beam3D(:,:,slicenum);
    
    upperlim = -FOV*[-sin(theta) cos(theta) 0] + iso;
    lowerlim = FOV*[-sin(theta) cos(theta) 0] + iso;
    
    [y,x] = ndgrid((1:ny)*imgres,(1:nx)*imgres);
    sign1 = sign(y-src(2)-(upperlim(2) - src(2))/(upperlim(1) - src(1))*((x - src(1))));
    sign2 = sign(y-src(2)-(lowerlim(2) - src(2))/(lowerlim(1) - src(1))*((x - src(1))));
    maskbeam = (sign1.*sign2<0)';
    if(nnz(maskbeam)>1e+04 && FOV<50)
        maskbeam = (sign1.*sign2>0)';
    end
    %     figure;imshow([dose1beam/max(dose1beam(:)) maskbeam])
    %     C = imfuse(dose1beam/max(dose1beam(:)),maskbeam,'falsecolor','Scaling','joint','ColorChannels',[1 2 0]);
    %     figure(49);imshow(C)
    %     set(gcf,'units','normalized','outerposition',[0 0 1 1]);
    
    fluence = fluence + img_fluence_filled(:,:,BeamNo).*maskbeam;
    figure(9);imshow(fluence,[])
end

fluence2 = fluence;
fluence2(mask0==0) = 0;
fluence2 = imgaussfilt(fluence2);
figure;imshow(fluence2,[])

% Ind = BeamletInd(:,:,beamlist);
% Ind = Ind(Ind>0);
% xf = zeros(numbeamlets,1);
% xf(Ind) = numevents;
% doseselectbeam3D = reshape(full(M*xf),size(masks{1}.mask));
% doseselectbeam = doseselectbeam3D(:,:,slicenum);
% doseselectbeam(mask0==0) = 0;
% figure;imshow(doseselectbeam*10,[]);colormap(jet);colorbar; set(gca,'FontSize',30)
%
% Anni1selectbeam3D = reshape(full(M_Anni*xf),size(masks{1}.mask));
% Anni1selectbeam = Anni1selectbeam3D(:,:,slicenum);
% figure;imshow(Anni1selectbeam,[])
%
% Anni1selectbeam_corrected = Anni1selectbeam./fluence2;
% Anni1selectbeam_corrected(mask0==0) = 0;
% Anni1selectbeam_corrected(fluence2<max(fluence2(:))*0.2) = 0;
% figure(10);imshow(Anni1selectbeam_corrected,[])

%% Compute beam paths
beamSizes = squeeze(sum(sum(params.BeamletLog0,1),2));
cumsumbeamSizes = cumsum([0; beamSizes]);
beamNoshift = cumsumbeamSizes(beamNo);
beamletIDs = double(beamletNo) + beamNoshift;

beamletpix = beamletwidth/imgres;
FOV = size(BeamletLog0,1)*beamletwidth;
beamletlist = 1:numbeamlets;
for iBeamlet = beamletlist
    [xlet,ylet,BeamNo] = ind2sub(size(BeamletInd),find(BeamletInd==iBeamlet));
    theta = mod(-beamangles(BeamNo) + 3.1436,2*pi);
    src = 1000*[cos(theta) sin(theta) 0] + iso;
    
    srcs(iBeamlet,:) = src - iso;
    beampathsy(iBeamlet,:) = (FOV/2-(xlet-1/2)*beamletwidth)*[-sin(theta) cos(theta) 0];
    
    %     xf = zeros(numbeamlets,1);
    %     xf(iBeamlet) = numevents;
    %     dose1beam3D = reshape(full(M*xf),size(masks{1}.mask));
    %     dose1beam = dose1beam3D(:,:,slicenum);
    %     Anni1beam3D = reshape(full(M_Anni*xf),size(masks{1}.mask));
    %     Anni1beam = Anni1beam3D(:,:,slicenum);
    %
    %     upperlim = (FOV/2-(xlet-1)*beamletwidth)*[-sin(theta) cos(theta) 0] + iso;
    %     lowerlim = (FOV/2-(xlet)*beamletwidth)*[-sin(theta) cos(theta) 0] + iso;
    %
    %     [y,x] = ndgrid((1:ny)*imgres,(1:nx)*imgres);
    %     sign1 = sign(y-src(2)-(upperlim(2) - src(2))/(upperlim(1) - src(1))*((x - src(1))));
    %     sign2 = sign(y-src(2)-(lowerlim(2) - src(2))/(lowerlim(1) - src(1))*((x - src(1))));
    %     maskbeam = (sign1.*sign2<=0)';
    %     if(nnz(maskbeam)>1e+04 && FOV<50)
    %         maskbeam = (sign1.*sign2>0)';
    %     end
    %     figure(100);imshow([dose1beam/max(dose1beam(:)) maskbeam])
    %     C = imfuse(dose1beam/max(dose1beam(:)),maskbeam,'falsecolor','Scaling','joint','ColorChannels',[1 2 0]);
    %     figure(49);imshow(C)
    
end

%% Image Reconstruction
EnergyResolution = 0.1;

R1 = 1200;
distrange = 300;
nb_cryst = max(detectorIds);
ig = image_geom('nx', size(img,1), 'ny', size(img,2), 'fov', size(img,1)*imgres);
reconparams = struct('nb_cryst',nb_cryst,'R1',R1,'distrange',distrange,...
    'imgres',imgres,'imgsize',ceil([ig.nx, ig.ny]));

CoincidenceTime = 1;

doserate = 0.1/60
detectorefficiency = 1
load(fullfile(dosematrixFolder,[patientName projectName '_CorrectedTime_' ...
    num2str(doserate*6000) 'MUpermin_detectorefficiency_' num2str(detectorefficiency) '.mat']),...
    'detectorefficiency','ImagingTime','CorrectedTime','eventrate')

Ind_coin_511 = IdentifyLOR_511(energy, CorrectedTime, CoincidenceTime);

%% Image Reconstruction FBP
TimeResolution = 0.4; % 400 ps
CorrectedTime_TR = CorrectedTime + TimeResolution*randn(size(CorrectedTime));
Ind_coin_accept = IdentifyLOR(energy, CorrectedTime_TR, CoincidenceTime, EnergyResolution);
TruePositive = length(intersect(Ind_coin_511(:,1).*Ind_coin_511(:,2),Ind_coin_accept(:,1).*Ind_coin_accept(:,2)))/length(Ind_coin_accept(:,1));

detid_pair = detectorIds(Ind_coin_accept);
[sino, dr, newunidist, sinobuff, unidist] = rebin_PET2(detid_pair, nb_cryst, R1, distrange);

sg = sino_geom('par', 'nb', size(sino,1), 'na', size(sino,2), 'dr', dr);

img_fbp_noattenuationcorrect = em_fbp_QL(sg, ig, sino);
ind1 = 0; ind2 = 0;
Anni2D = TranslateFigure(Anni3D(:,:,slicenum),ind1,ind2);
% figure;imshow([Anni2D/max(Anni2D(:)),img_fbp_noattenuationcorrect/max(img_fbp_noattenuationcorrect(:))],[])
% C = imfuse(Anni2D/max(Anni2D(:)),img_fbp_noattenuationcorrect/max(img_fbp_noattenuationcorrect(:)),'falsecolor','Scaling','joint','ColorChannels',[1 2 0]);
% figure; imshow(C)
mumapnew = TranslateFigure(mumap,ind1,ind2);
G = Gtomo2_strip(sg, ig);
li = G * mumapnew;
ci = exp(-li);

img_fbp = em_fbp_QL(sg, ig, sino./ci);
img_fbp_corrected = img_fbp./fluence2;
img_fbp_corrected(mask0==0)=0;
figure;imshow([img_fbp_corrected],[])


ForBack.applyFP = @(x) G*x;
ForBack.applyBP = @(x) G'*x;
gamma = 500;
mu = 1e-05;
[x_TV, Maincost_TV] = IterRecon_PairProd_TV_FISTA (ForBack, sino./ci, gamma, mu, [ig.nx, ig.ny]);

img_TV_corrected = x_TV./fluence2;
img_TV_corrected(mask0==0)=0;
figure;imshow([img_fbp_corrected img_TV_corrected],[])







