clear
close all
clc

% t = tic;
% while toc(t)<7200
%     pause(2)
% end

projectName = 'PairProd';
patientName = 'GBMHY_final_100m';
taglist = {'run01','run02','run03','run04','run05','run06','run07','run08','run09','run10'};
MergedName = 'run01torun10_1000MUpermin';

M0 = 0;
M_Anni0 = 0;
detectorIds0 = [];
beamNo0 = [];
beamletNo0 = [];
energy0 = [];
eventIds0 = [];
globalTimes0 = [];
CorrectedTime0 = [];
beamletIDs0 = [];
ImagingTime0 = 0;
deltatime = 10;

for nruns = 1:numel(taglist)
    tag = taglist{nruns};
    
    patFolder = fullfile('/media/raid0/qlyu/PairProd/datatest',patientName);
    projectFolder = fullfile(patFolder,projectName);
    dosematrixFolder = fullfile(projectFolder,'dosematrix');
    
    load(fullfile(dosematrixFolder,[patientName projectName '_M_HighRes_' tag '.mat']),'M','M_Anni','dose_data','masks');
    
    load(fullfile(dosematrixFolder,[patientName projectName '_ringdetection_1000MUpermin_' tag '.mat']),...
        'detectorIds','beamNo','beamletNo','energy','eventIds','globalTimes','beamletIDs','ImagingTime','CorrectedTime','numeventsvec');
    
    if(any(find(sum(M)')-find(numeventsvec)))
        error('Simulation error !!!')
    end
    
    if(nruns == 1)
        numeventsvec0 = zeros(size(numeventsvec));
        timeshift = 0;
        M0 = M;
        M_Anni0 = M_Anni;
    else
        timeshift = max(CorrectedTime0(:))+deltatime;
        M0 = (M0.*numeventsvec0' + M.*numeventsvec')./(numeventsvec' + numeventsvec0');
        M_Anni0 = (M_Anni0.*numeventsvec0' + M_Anni.*numeventsvec')./(numeventsvec' + numeventsvec0');
        M0(:,numeventsvec0==0) = 0;
        M_Anni0(:,numeventsvec0==0) = 0;
    end
    detectorIds0 = [detectorIds0;detectorIds];
    beamNo0 = [beamNo0;beamNo];
    beamletNo0 = [beamletNo0;beamletNo];
    beamletIDs0 = [beamletIDs0;beamletIDs];
    energy0 = [energy0;energy];
    eventIds = eventIds + numeventsvec0(beamletNo);
    eventIds0 = [eventIds0;eventIds];
    globalTimes0 = [globalTimes0;globalTimes];
    CorrectedTime0 = [CorrectedTime0; CorrectedTime + timeshift]; 
    ImagingTime0 = ImagingTime0 + ImagingTime;
    numeventsvec0 = numeventsvec0 + numeventsvec;    
end


M = M0;
M_Anni = M_Anni0;
detectorIds = detectorIds0;
beamNo = beamNo0;
beamletNo = beamletNo0;
energy = energy0;
eventIds = eventIds0;
globalTimes = globalTimes0;
numeventsvec = numeventsvec0;
CorrectedTime = CorrectedTime0;
ImagingTime = ImagingTime0;
beamletIDs = beamletIDs0;

paramsFolder = fullfile(projectFolder,'params');
InfoNum = 0;
load(fullfile(paramsFolder,['StructureInfo' num2str(InfoNum) '.mat']),'StructureInfo');
load(fullfile(dosematrixFolder,[patientName projectName '_dicomimg.mat']),'img','imgres');
ParamsNum = 0;
load(fullfile(paramsFolder,['params' num2str(ParamsNum) '.mat']),'params');


%% Save files
patientName = [patientName, '_', MergedName];
patFolder = fullfile('/media/raid0/qlyu/PairProd/datatest',patientName);
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
save(fullfile(dosematrixFolder,[patientName projectName '_M_HighRes.mat']),'M','M_Anni','dose_data','masks','-v7.3');


save(fullfile(dosematrixFolder,[patientName projectName '_ringdetection_directmerge.mat']),...
    'detectorIds','beamNo','beamletNo','energy','eventIds','globalTimes','beamletIDs','ImagingTime','CorrectedTime','numeventsvec','-v7.3');


% load(fullfile(dosematrixFolder,[patientName projectName '_ringdetection.mat']),'eventrate');
% save(fullfile(dosematrixFolder,[patientName projectName '_ringdetection.mat']),...
%     'detectorIds','beamNo','beamletNo','energy','eventIds','globalTimes','numeventsvec');
% 
% beamSizes = squeeze(sum(sum(params.BeamletLog0,1),2));
% cumsumbeamSizes = cumsum([0; beamSizes]);
% beamNoshift = cumsumbeamSizes(beamNo);
% beamletIDs = double(beamletNo) + beamNoshift;
% 
% %% fluence map segments
% numbeamlets = size(M,2);
% 
% x_ = numeventsvec;
% dose = reshape(M*x_,size(masks{1}.mask));
% figure;imshow3D(dose,[])
% 
% %% Time correction: Adding time of previous events
% numeventbatch = 1e+06;
% numevent = max(eventIds);
% numbatches = ceil(numevent/numeventbatch);
% deltatime_event = normrnd(eventrate,eventrate/5,numeventbatch,numbeamlets);
% cumsum_eventtime_batch = cumsum(deltatime_event);
% 
% event_perbatchIDs = mod(eventIds-1, numeventbatch)+1;
% event_batchID = (eventIds - event_perbatchIDs)/numeventbatch + 1;
% event_perbatch_beamletIDs = sub2ind([numeventbatch,numbeamlets], event_perbatchIDs, beamletIDs);
% 
% numbeams = max(beamNo(:));
% batchtime = max(cumsum_eventtime_batch(event_perbatch_beamletIDs));
% beamtime = batchtime*numbatches;
% deltatime_beam = beamtime*(1:numbeams)'; % 1 ms
% 
% CorrectedTime = globalTimes + cumsum_eventtime_batch(event_perbatch_beamletIDs)...
%     + event_batchID*batchtime + beamNo*beamtime;
% [sortedtime, sortInd] = sort(CorrectedTime);
% save(fullfile(dosematrixFolder,[patientName projectName '_ringdetection.mat']),'CorrectedTime','sortedtime','sortInd','-append');
% 






