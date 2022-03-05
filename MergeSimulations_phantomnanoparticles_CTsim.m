clear
close all
clc

% t = tic;
% while toc(t)<7200
%     pause(2)
% end

Folder = '/media/raid1/qlyu/PairProd/datatest';
projectName = 'CTsim';
patientNameList = {'phantom_nanoparticles_360beam_200m_slice2cm_CTsimNEW',...
    'phantom_nanoparticles_360beam_200m_slice2cm_CTsimNEW_run2',...
    'phantom_nanoparticles_360beam_200m_slice2cm_CTsimNEW_run3',...
    'phantom_nanoparticles_360beam_200m_slice2cm_CTsimNEW_run4',...
    'phantom_nanoparticles_360beam_200m_slice2cm_CTsimNEW_run5'};
MergedName = 'phantom_nanoparticles_360beam_1b_slice2cm_merged';
numparticlesList = [200,200,200,200,200]*1e+06;

blankfieldName = 'phantom_nanoparticles_360beam_200m_slice2cm_CTsimNEW_blankfield';
numparticles_blankfield = 2e+08;
load(fullfile(Folder,blankfieldName,'dosecalc','CTsim_CTprojection.mat'),'CTprojection');
CTprojection_blankfield = CTprojection(:,:,1);
CTprojectionNew = CTprojection*numparticles_blankfield;
if(max(CTprojectionNew - round(CTprojectionNew),[],'all')>0.01 || max(CTprojectionNew/2 - round(CTprojectionNew/2),[],'all')<0.01)
    error('Number of Particles mismatch!!!')
end

M0 = 0;
CTprojection0 = 0;
numparticles0 = 0;
for ii = 1:numel(patientNameList)
    patientName = patientNameList{ii};
    numparticles = numparticlesList(ii);
    
    patFolder = fullfile(Folder,patientName);
    projectFolder = fullfile(patFolder,projectName);
    dosematrixFolder = fullfile(projectFolder,'dosematrix');
    dosecalcFolder = fullfile(patFolder,'dosecalc');
    
    load(fullfile(dosematrixFolder,[patientName projectName '_M_HighRes.mat']),'M','dose_data','masks');
    load(fullfile(dosecalcFolder,[projectName '_CTprojection.mat']),'CTprojection');
    CTprojectionNew = CTprojection*numparticles;
    
    if(max(CTprojectionNew - round(CTprojectionNew),[],'all')>0.01 || max(CTprojectionNew/2 - round(CTprojectionNew/2),[],'all')<0.01)
        error('Number of Particles mismatch!!!')
    end
    
    if(ii == 1)
        M0 = M;
    else
        M0 = (M0.*numparticles0 + M.*numparticles)./(numparticles0 + numparticles);
    end
    numparticles0 = numparticles0 + numparticles;
    CTprojection0 = CTprojection0 + CTprojectionNew;
end


M = M0;
numparticles = numparticles0;
CTprojection = CTprojection0/numparticles0;

LI = log(permute(repmat(CTprojection_blankfield,[1,1,size(CTprojection,3)]),[2,1,3])./permute(CTprojection,[2,1,3]));
LI(isinf(LI)) = 0;
LI(isnan(LI)) = 0;
figure;imshow3D(LI,[]);


%%
paramsFolder = fullfile(projectFolder,'params');
InfoNum = 0;
load(fullfile(paramsFolder,['StructureInfo' num2str(InfoNum) '.mat']),'StructureInfo');
load(fullfile(dosematrixFolder,[patientName projectName '_dicomimg.mat']),'img','imgres');
ParamsNum = 0;
load(fullfile(paramsFolder,['params' num2str(ParamsNum) '.mat']),'params');

%% Image Reconstruction
cg = ct_geom('fan', ...
    'ns', 250, ... % detector channels
    'nt', 500, ... % detector rows
    'na', 360, ... % angular samples
    'offset_s', 0, ... % quarter-detector offset
    'dsd', 1000, ...
    'dod', 333.3, ...
    'dfs', inf, ... % arc
    'ds', 2, ... % detector pitch
    'dt', 2, ... % detector row spacing for 0.625mm slices, 2009-12-06
    'pitch',0,...
    'orbit_start',90);
ig = image_geom('nx', size(img,1), 'ny', size(img,2), 'nz', 100, 'fov', size(img,1)*imgres);
mask2 = true([ig.nx ig.ny]);
mask2(end) = 0; % trick: test it
ig.mask = repmat(mask2, [1 1 ig.nz]);
li_hat = fdk_filter(LI(126:end-125,:,1:end), 'ramp', cg.dsd, cg.dfs, cg.ds);

args = {flip(li_hat,1), cg, ig, 'ia_skip', 1}; % increase 1 for faster debugging
CT_FBP = cbct_back(args{:}, 'use_mex', 1, 'back_call', @jf_mex);
figure;imshow3D(CT_FBP,[0,0.15])

%% Save files
patientName = MergedName;
patFolder = fullfile('/media/raid1/qlyu/PairProd/datatest',patientName);
projectFolder = fullfile(patFolder,projectName);
paramsFolder = fullfile(projectFolder,'params');
dosematrixFolder = fullfile(projectFolder,'dosematrix');

mkdir(dosematrixFolder)
mkdir(paramsFolder)

InfoNum = 0;
save(fullfile(paramsFolder,['StructureInfo' num2str(InfoNum) '.mat']),'StructureInfo');
save(fullfile(dosematrixFolder,[patientName projectName '_dicomimg.mat']),'img','imgres');
ParamsNum = 0;
save(fullfile(paramsFolder,['params' num2str(ParamsNum) '.mat']),'params');

dosematrixFolder = fullfile(projectFolder,'dosematrix');
save(fullfile(dosematrixFolder,[patientName projectName '_M_HighRes.mat']),'M','dose_data','masks','-v7.3');

save(fullfile(dosematrixFolder,'LineIntegrals.mat'),'CTprojection','CTprojection_blankfield','LI','numparticles');

CT_Dose = reshape(M*ones(size(M,2),1)*numparticles,size(StructureInfo(1).Mask));

resultsFolder = fullfile(projectFolder,'results');
mkdir(resultsFolder)
save(fullfile(resultsFolder,'Recon_CT.mat'),'CT_FBP','CT_Dose')

figure;imagesc(CT_Dose(:,:,ceil(end/2)));colorbar;colormap(jet);
axis off; axis equal;
saveas(gcf,fullfile(resultsFolder,'CT_Dose.png'))

figure;imshow(CT_FBP(:,:,ceil(end/2)),[0,0.15])
saveas(gcf,fullfile(resultsFolder,'CT_FBP.png'))

