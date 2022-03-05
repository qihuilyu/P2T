clear
close all
clc

projectName = 'PairProd';
patientNameList = {'PartialViewPhantom_LimitedROI_2mmbeamlet_30m',...
    'PartialViewPhantom_LimitedROI_2mmbeamlet_100m','PartialViewPhantom_LimitedROI_2mmbeamlet_100m_run2'};
MergedName = 'PartialViewPhantom_LimitedROI_2mmbeamlet_230m_merged';

M0 = 0;
M_Anni0 = 0;
detectorIds0 = [];
beamNo0 = [];
beamletNo0 = [];
energy0 = [];
eventIds0 = [];
globalTimes0 = [];
numevent0 = 0;


for ii = 1:numel(patientNameList)
    patientName = patientNameList{ii};
    patFolder = fullfile('/media/raid1/qlyu/PairProd/datatest',patientName);
    projectFolder = fullfile(patFolder,projectName);
    paramsFolder = fullfile(projectFolder,'params');
    dosematrixFolder = fullfile(projectFolder,'dosematrix');
    
    InfoNum = 0;
    load(fullfile(paramsFolder,['StructureInfo' num2str(InfoNum) '.mat']),'StructureInfo');
    load(fullfile(dosematrixFolder,[patientName projectName '_dicomimg.mat']),'img','imgres');
    ParamsNum = 0;
    load(fullfile(paramsFolder,['params' num2str(ParamsNum) '.mat']),'params');
    
    load(fullfile(dosematrixFolder,[patientName projectName '_M_HighRes.mat']),'M','M_Anni','dose_data','masks');
    
    load(fullfile(dosematrixFolder,[patientName projectName '_ringdetection.mat']),...
        'detectorIds','beamNo','beamletNo','energy','eventIds','globalTimes',...
        'numevent');
    numevent = round(numevent/100)*100;
    M0 = (M0*numevent0 + M*numevent)/(numevent + numevent0);
    M_Anni0 = (M_Anni0*numevent0 + M_Anni*numevent)/(numevent + numevent0);
    detectorIds0 = [detectorIds0;detectorIds];
    beamNo0 = [beamNo0;beamNo];
    beamletNo0 = [beamletNo0;beamletNo];
    energy0 = [energy0;energy];
    eventIds = eventIds + numevent0;
    eventIds0 = [eventIds0;eventIds];
    globalTimes0 = [globalTimes0;globalTimes];
    numevent0 = numevent0 + numevent;
    
    
end


M = M0;
M_Anni = M_Anni0;
detectorIds = detectorIds0;
beamNo = beamNo0;
beamletNo = beamletNo0;
energy = energy0;
eventIds = eventIds0;
globalTimes = globalTimes0;
numevent = numevent0;


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
save(fullfile(dosematrixFolder,[patientName projectName '_M_HighRes.mat']),'M','M_Anni','dose_data','masks','-v7.3');

save(fullfile(dosematrixFolder,[patientName projectName '_ringdetection.mat']),...
    'detectorIds','beamNo','beamletNo','energy','eventIds','globalTimes',...
    'numevent');

%% fluence map segments
beamSizes = squeeze(sum(sum(params.BeamletLog0,1),2));
cumsumbeamSizes = cumsum([0; beamSizes]);
beamNoshift = cumsumbeamSizes(beamNo);
beamletIDs = double(beamletNo) + beamNoshift;
numbeamlets = size(M,2);

x_ = ones(numbeamlets,1);
dose = reshape(M*x_,size(masks{1}.mask));
figure;imshow3D(dose,[])

selectbeamNo = 1;
x_onebeam = zeros(numbeamlets,1);
x_onebeam(cumsumbeamSizes(selectbeamNo)+1:cumsumbeamSizes(selectbeamNo+1)) = 1;
dose_onebeam = reshape(M*x_onebeam,size(masks{1}.mask));
figure;imshow3D(dose_onebeam,[])

clearvars M M_Anni


%% Time correction: Adding time of previous events
for doserate = [0.1/60 1/60 10/60]
    for detectorefficiency = [0.1 1]
        time = max(dose_onebeam(:))*numevent/doserate;
        eventrate = time/numevent*1e+09/detectorefficiency; % ns/event
        
        numeventbatch = 1e+06;
        numbatches = ceil(numevent/numeventbatch);
        deltatime_event = rand(numeventbatch,numbeamlets)*eventrate*2;
        cumsum_eventtime_batch = cumsum(deltatime_event);
        
        event_perbatchIDs = mod(eventIds-1, numeventbatch)+1;
        event_batchID = (eventIds - event_perbatchIDs)/numeventbatch + 1;
        event_perbatch_beamletIDs = sub2ind([numeventbatch,numbeamlets], event_perbatchIDs, beamletIDs);
        
        batchtime = max(cumsum_eventtime_batch(event_perbatch_beamletIDs));
        beamtime = batchtime*numbatches;
        
        CorrectedTime = globalTimes + cumsum_eventtime_batch(event_perbatch_beamletIDs)...
            + (event_batchID-1)*batchtime + (beamNo-1)*beamtime;
        [sortedtime, sortInd] = sort(CorrectedTime);
        ImagingTime = max(CorrectedTime)/1e+09;  % s
        save(fullfile(dosematrixFolder,[patientName projectName '_CorrectedTime_' ...
            num2str(doserate*6000) 'MUpermin_detectorefficiency_' num2str(detectorefficiency) '.mat']),...
            'detectorefficiency','ImagingTime','CorrectedTime','sortedtime','sortInd','eventrate','-v7.3');
    end
end

%% Time correction beamlet by beamlet
eventrate = 2; % ns/event

deltatime_event = rand(numevent,1)*eventrate*2;
cumsum_eventtime = cumsum(deltatime_event);
time_perbeamlet = eventrate*numevent + 20;

CorrectedTime = globalTimes + cumsum_eventtime(eventIds)...
        + (beamletIDs-1)*time_perbeamlet;
[sortedtime, sortInd] = sort(CorrectedTime);
ImagingTime = max(CorrectedTime)/1e+09;  % s
save(fullfile(dosematrixFolder,[patientName projectName '_CorrectedTime_perbeamletdelivery_eventrate_' num2str(eventrate) '.mat']),...
    'detectorefficiency','ImagingTime','CorrectedTime','sortedtime','sortInd','eventrate','-v7.3');



