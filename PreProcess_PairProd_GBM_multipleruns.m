clear
close all
clc

% t = tic;
% while toc(t)<7200
%     pause(2)
% end
% !('/media/raid1/qlyu/PairProd/datatest/collect_doescalc_pairprod.sh')

patientName = 'GBMHY_final_100m';
projectName = 'PairProd';
patFolder = fullfile('/media/raid0/qlyu/PairProd/datatest',patientName);
dosecalcFolder = fullfile(patFolder,'dosecalc');
maskfile = fullfile(dosecalcFolder,'PairProd_masks.h5');
fmapsfile = fullfile(dosecalcFolder,'PairProd_fmaps.h5');
numeventsfile = fullfile(dosecalcFolder,[patientName '_nruns_10.txt']);
taglist = {'run01','run02','run03','run04','run05','run06','run07','run08','run09','run10'};

projectFolder = fullfile(patFolder,'PairProd');
paramsFolder = fullfile(projectFolder,'params');
mkdir(paramsFolder)
dosematrixFolder = fullfile(projectFolder,'dosematrix');
mkdir(dosematrixFolder)

%% Dicom
DicomPath = fullfile(dosecalcFolder,'ctdata');
baseFileNames = dir([DicomPath '/*.dcm']);
[sortedFile,index] = sort_nat({baseFileNames.name});

img = [];
for ii = 1:numel(sortedFile)
    dcinfo = dicominfo(fullfile(DicomPath,sortedFile{ii}));
    if(~strcmpi(dcinfo.Modality,'RTstruct'))
        imgres = dcinfo.SliceThickness;
        img(:,:,ii) = dicomread(fullfile(DicomPath,sortedFile{ii}));
    end
end
figure;imshow3D(img,[0,2000])
save(fullfile(dosematrixFolder,[patientName projectName '_dicomimg.mat']),'img','imgres');

%% numevents
fileID = fopen(numeventsfile,'r');
formatSpec = '%d';
numeventsvec = fscanf(fileID,formatSpec);
fclose(fileID);

%% Multiple runs
for itag = 1:numel(taglist)
    tag = taglist{itag};
    h5file = fullfile(dosecalcFolder,['PairProd_beamletdose_' tag '.h5']);
    Anni3Dfile = fullfile(dosecalcFolder,['PairProd_NofPositronAnni3D_' tag '.h5']);
    DetectedEventsfile = fullfile(dosecalcFolder,['PairProd_DetectedEvents_' tag '.h5']);
    
    %% masks, fmaps, dose matrix, annihilation matrix
    [M,M_Anni,dose_data,masks]=BuildDoseMatrix_PairProd(h5file, maskfile, fmapsfile, Anni3Dfile);
    
    dose = reshape(full(sum(M,2)),size(masks{1}.mask));
    Anni3D = reshape(full(sum(M_Anni,2)),size(masks{1}.mask));
    
    figure;imshow3D(dose)
    figure;imshow3D(Anni3D)
    
    pdose = 25;
    [StructureInfo, params] = InitIMRTparams_DLMCforRyan(M,dose_data,masks,pdose,[1,2,0]);
    
    ParamsNum = 0;
    save(fullfile(paramsFolder,['params' num2str(ParamsNum) '.mat']),'params');
    InfoNum = 0;
    save(fullfile(paramsFolder,['StructureInfo' num2str(InfoNum) '.mat']),'StructureInfo');
    
    save(fullfile(dosematrixFolder,[patientName projectName '_M_HighRes_' tag '.mat']),'M','M_Anni','dose_data','masks','-v7.3');
    
    
    %% Ring detection
    info = h5info(DetectedEventsfile);
    detectorIds = double(h5read(DetectedEventsfile, '/detectorIds')) + 1; %
    beamNo = double(h5read(DetectedEventsfile, '/beamNo')) + 1; %
    beamletNo = double(h5read(DetectedEventsfile, '/beamletNo')) + 1; %
    energy = h5read(DetectedEventsfile, '/energy'); %
    eventIds = double(h5read(DetectedEventsfile, '/eventIds')) + 1; %
    globalTimes = h5read(DetectedEventsfile, '/globalTimes'); %
    Mega = [globalTimes,eventIds,energy,beamletNo,beamNo,detectorIds];
    save(fullfile(dosematrixFolder,[patientName projectName '_ringdetection_original_' tag '.mat']),'detectorIds','beamNo','beamletNo','energy','eventIds','globalTimes','-v7.3');
    
    %% Clean data
    numevent = max(eventIds);
    numevent = numevent + 99 - mod(numevent-1,100);
    beamSizes = squeeze(sum(sum(params.BeamletLog0,1),2));
    cumsumbeamSizes = cumsum([0; beamSizes]);
    beamNoshift = cumsumbeamSizes(beamNo);
    beamletIDs = double(beamletNo) + beamNoshift;
    AlleventID = (beamletIDs-1)*numevent + eventIds;
    nb_cryst = max(detectorIds);
    AlldetectorID = (AlleventID-1)*nb_cryst + detectorIds;
    [sortedAlldetectorID, sortAlldetectorIDInd] = sort(AlldetectorID);
    sortInd_sameparticle = find(diff(sortedAlldetectorID)==1);
    Ind_coin1 = sortAlldetectorIDInd(sortInd_sameparticle);
    Ind_coin2 = sortAlldetectorIDInd(sortInd_sameparticle+1);
    mask_sameenergy = (energy(Ind_coin1)-energy(Ind_coin2)==0);
    timediff = globalTimes(Ind_coin1)-globalTimes(Ind_coin2);
    badIDbuff1 = Ind_coin1(mask_sameenergy & timediff>0);
    badIDbuff2 = Ind_coin2(mask_sameenergy & timediff<0);
    badID1 = union(badIDbuff1,badIDbuff2);
    
    clearvars Ind_coin1 Ind_coin2 mask_sameenergy sortInd_sameparticle sortAlldetectorIDInd sortedAlldetectorID timediff badIDbuff1 badIDbuff2
    
    % boundary issue(det id: 1 and 1440)
    AlldetectorID2 = (AlleventID-1)*nb_cryst + mod(detectorIds,nb_cryst);
    [sortedAlldetectorID, sortAlldetectorIDInd] = sort(AlldetectorID2);
    sortInd_sameparticle = find(diff(sortedAlldetectorID)==1);
    Ind_coin1 = sortAlldetectorIDInd(sortInd_sameparticle);
    Ind_coin2 = sortAlldetectorIDInd(sortInd_sameparticle+1);
    mask_sameenergy = (energy(Ind_coin1)-energy(Ind_coin2)==0);
    timediff = globalTimes(Ind_coin1)-globalTimes(Ind_coin2);
    badIDbuff1 = Ind_coin1(mask_sameenergy & timediff>0);
    badIDbuff2 = Ind_coin2(mask_sameenergy & timediff<0);
    badID2 = union(badIDbuff1,badIDbuff2);
    
    badID = union(badID1,badID2);
    clearvars Ind_coin1 Ind_coin2 mask_sameenergy sortInd_sameparticle sortAlldetectorIDInd sortedAlldetectorID timediff badIDbuff1 badIDbuff2 badID1 badID2
    
    beamletIDs(badID,:) = [];
    detectorIds(badID,:) = [];
    beamNo(badID,:) = [];
    beamletNo(badID,:) = [];
    energy(badID,:) = [];
    eventIds(badID,:) = [];
    globalTimes(badID,:) = [];
    Mega(badID,:) = [];
    % save(fullfile(dosematrixFolder,[patientName projectName '_ringdetection_' tag '.mat']),'detectorIds','beamNo','beamletNo','energy','eventIds','globalTimes','-v7.3');
    
    %% Reject detectors associated with primary beam; select photons based on energy resolution
    
    EnergyResolution = 0.1;
    Ind_accept = find(abs(energy-0.511)<0.511*EnergyResolution);
    
    beamletIDs = beamletIDs(Ind_accept);
    detectorIds = detectorIds(Ind_accept);
    beamNo = beamNo(Ind_accept);
    beamletNo = beamletNo(Ind_accept);
    energy = energy(Ind_accept);
    eventIds = eventIds(Ind_accept);
    globalTimes = globalTimes(Ind_accept);
    Mega = Mega(Ind_accept,:);
    
    beamdet_accept = full(sparse(beamNo,detectorIds,1));
    figure; set(gcf,'pos',[2715   148    1480    1001])
    for ii = 1:size(beamdet_accept,1)
        plot(beamdet_accept(ii,:),'LineWidth',2); hold on
    end
    % legend({'beam 1','beam 5','beam 9','beam 13','beam 17'})
%     legend({'beam 1','beam 3','beam 5','beam 7'})
    xlabel('Detector ID')
    ylabel('Detected photon counts')
    title('Detected photon counts within 10% energy resolution')
    set(gca,'FontSize',20)
    saveas(gcf,fullfile(dosematrixFolder,['Original_counts_' tag '.png']))
    
    thresh = median(beamdet_accept,2)*10;
    beamNo_detectorIDs_rej = find(beamdet_accept-repmat(thresh,[1,size(beamdet_accept,2)])>0);
    beamNo_detectorIDs = (detectorIds-1)*size(beamdet_accept,1) + beamNo;
    badIDmask = ismember(beamNo_detectorIDs,beamNo_detectorIDs_rej);
    
    newbeamdet_accept = full(sparse(beamNo(badIDmask==0),detectorIds(badIDmask==0),1));
    figure; set(gcf,'pos',[2715   148    1480    1001])
    for ii = 1:size(newbeamdet_accept,1)
        plot(newbeamdet_accept(ii,:),'LineWidth',2); hold on
    end
    % legend({'beam 1','beam 5','beam 9','beam 13','beam 17'})
%     legend({'beam 1','beam 3','beam 5','beam 7'})
    xlabel('Detector ID')
    ylabel('Detected photon counts')
    title('Detected photon counts within 10% energy resolution')
    set(gca,'FontSize',20)
    saveas(gcf,fullfile(dosematrixFolder,['Processed_counts_' tag '.png']))
    
    beamletIDs(badIDmask,:) = [];
    detectorIds(badIDmask,:) = [];
    beamNo(badIDmask,:) = [];
    beamletNo(badIDmask,:) = [];
    energy(badIDmask,:) = [];
    eventIds(badIDmask,:) = [];
    globalTimes(badIDmask,:) = [];
    Mega(badIDmask,:) = [];
    
    
    %% fluence map segments
    numbeamlets = size(M,2);
    
    x_ = numeventsvec;
    dose = reshape(M*x_,size(masks{1}.mask));
    figure;imshow3D(dose,[])
    
    
    %% Time correction: Adding time of previous events
    load(fullfile(dosematrixFolder,[patientName projectName '_ringdetection_100MUpermin.mat']),'eventrate');
    
    numeventbatch = 1e+06;
    numbatches = ceil(numevent/numeventbatch);
    deltatime_event = rand(numeventbatch,numbeamlets)*eventrate*2;
    cumsum_eventtime_batch = cumsum(deltatime_event);
    
    event_perbatchIDs = mod(eventIds-1, numeventbatch)+1;
    event_batchID = (eventIds - event_perbatchIDs)/numeventbatch + 1;
    event_perbatch_beamletIDs = sub2ind([numeventbatch,numbeamlets], event_perbatchIDs, beamletIDs);
    
    numbeams = max(beamNo(:));
    batchtime = max(cumsum_eventtime_batch(event_perbatch_beamletIDs));
    beamtime = batchtime*numbatches;
    
    CorrectedTime = globalTimes + cumsum_eventtime_batch(event_perbatch_beamletIDs)...
        + (event_batchID-1)*batchtime + (beamNo-1)*beamtime;
    [sortedtime, sortInd] = sort(CorrectedTime);
    ImagingTime = max(CorrectedTime)/1e+09;  % s
    save(fullfile(dosematrixFolder,[patientName projectName '_ringdetection_100MUpermin_' tag '.mat']),...
        'detectorIds','beamNo','beamletNo','energy','eventIds','globalTimes','beamletIDs','ImagingTime',....
        'CorrectedTime','sortedtime','sortInd','numeventsvec','eventrate','-v7.3');
    
    %% Time correction: Adding time of previous events
    load(fullfile(dosematrixFolder,[patientName projectName '_ringdetection_1000MUpermin.mat']),'eventrate');
    
    numeventbatch = 1e+06;
    numbatches = ceil(numevent/numeventbatch);
    deltatime_event = rand(numeventbatch,numbeamlets)*eventrate*2;
    cumsum_eventtime_batch = cumsum(deltatime_event);
    
    event_perbatchIDs = mod(eventIds-1, numeventbatch)+1;
    event_batchID = (eventIds - event_perbatchIDs)/numeventbatch + 1;
    event_perbatch_beamletIDs = sub2ind([numeventbatch,numbeamlets], event_perbatchIDs, beamletIDs);
    
    numbeams = max(beamNo(:));
    batchtime = max(cumsum_eventtime_batch(event_perbatch_beamletIDs));
    beamtime = batchtime*numbatches;
    
    CorrectedTime = globalTimes + cumsum_eventtime_batch(event_perbatch_beamletIDs)...
        + (event_batchID-1)*batchtime + (beamNo-1)*beamtime;
    [sortedtime, sortInd] = sort(CorrectedTime);
    ImagingTime = max(CorrectedTime)/1e+09;  % s
    save(fullfile(dosematrixFolder,[patientName projectName '_ringdetection_1000MUpermin_' tag '.mat']),...
        'detectorIds','beamNo','beamletNo','energy','eventIds','globalTimes','beamletIDs','ImagingTime',....
        'CorrectedTime','sortedtime','sortInd','numeventsvec','eventrate','-v7.3');
        
end