clear
close all
clc
% 
% t = tic;
% while toc(t)<7200*3
%     pause(2)
% end
% !('/media/raid1/qlyu/PairProd/datatest/collect_doescalc_pairprod.sh')

patientName = 'CTphantom_experiment_5mmbeamlet_10m'; % 
projectName = 'PairProd';
patFolder = fullfile('/media/raid1/qlyu/PairProd/datatest',patientName);
dosecalcFolder = fullfile(patFolder,'dosecalc');
h5file = fullfile(dosecalcFolder,'PairProd_beamletdose.h5');
maskfile = fullfile(dosecalcFolder,'PairProd_masks.h5');
fmapsfile = fullfile(dosecalcFolder,'PairProd_fmaps.h5');
Anni3Dfile = fullfile(dosecalcFolder,'PairProd_NofPositronAnni3D.h5');
DetectedEventsfile = fullfile(dosecalcFolder,'PairProd_DetectedEvents.h5');

%% masks, fmaps, dose matrix, annihilation matrix
[M,M_Anni,dose_data,masks]=BuildDoseMatrix_PairProd(h5file, maskfile, fmapsfile, Anni3Dfile);  

dose = reshape(full(sum(M,2)),size(masks{1}.mask));
Anni3D = reshape(full(sum(M_Anni,2)),size(masks{1}.mask));

figure;imshow3D(dose)
figure;imshow3D(Anni3D)

pdose = 25;
[StructureInfo, params] = InitIMRTparams_DLMCforRyan(M,dose_data,masks,pdose,[1,2,0]);

projectFolder = fullfile(patFolder,'PairProd');
paramsFolder = fullfile(projectFolder,'params');
mkdir(paramsFolder)

ParamsNum = 0;
save(fullfile(paramsFolder,['params' num2str(ParamsNum) '.mat']),'params');
InfoNum = 0;
save(fullfile(paramsFolder,['StructureInfo' num2str(InfoNum) '.mat']),'StructureInfo');

dosematrixFolder = fullfile(projectFolder,'dosematrix');
mkdir(dosematrixFolder)
% save(fullfile(dosematrixFolder,[patientName projectName '_M_HighRes.mat']),'M','M_Anni','dose_data','masks','-v7.3');

%% fluence map segments
beamSizes = squeeze(sum(sum(params.BeamletLog0,1),2));
cumsumbeamSizes = cumsum([0; beamSizes]);
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


%% Ring detection
info = h5info(DetectedEventsfile);
detectorIds = double(h5read(DetectedEventsfile, '/detectorIds')) + 1; %
beamNo = double(h5read(DetectedEventsfile, '/beamNo')) + 1; %
beamletNo = double(h5read(DetectedEventsfile, '/beamletNo')) + 1; %
energy = h5read(DetectedEventsfile, '/energy'); %
eventIds = double(h5read(DetectedEventsfile, '/eventIds')) + 1; %
globalTimes = h5read(DetectedEventsfile, '/globalTimes'); %
% Mega = [globalTimes,eventIds,energy,beamletNo,beamNo,detectorIds];
% save(fullfile(dosematrixFolder,[patientName projectName '_ringdetection_original.mat']),'detectorIds','beamNo','beamletNo','energy','eventIds','globalTimes','-v7.3');

%% Clean data
numevent = max(eventIds);
numevent = numevent + 99 - mod(numevent-1,100);

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

clearvars AlldetectorID beamNoshift
clearvars Ind_coin1 Ind_coin2 mask_sameenergy sortInd_sameparticle sortAlldetectorIDInd sortedAlldetectorID timediff 
badID1 = union(badIDbuff1,badIDbuff2);
clearvars badIDbuff1 badIDbuff2

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

clearvars Ind_coin1 Ind_coin2 mask_sameenergy sortInd_sameparticle sortAlldetectorIDInd sortedAlldetectorID timediff
badID2 = union(badIDbuff1,badIDbuff2);
clearvars badIDbuff1 badIDbuff2
badID = union(badID1,badID2);
clearvars badID1 badID2

beamletIDs(badID,:) = [];
detectorIds(badID,:) = [];
beamNo(badID,:) = [];
beamletNo(badID,:) = [];
energy(badID,:) = [];
eventIds(badID,:) = [];
globalTimes(badID,:) = [];
% Mega(badID,:) = [];
% save(fullfile(dosematrixFolder,[patientName projectName '_ringdetection.mat']),'detectorIds','beamNo','beamletNo','energy','eventIds','globalTimes','-v7.3');

%% Reject detectors associated with primary beam; select photons based on energy resolution
beamdet_accept = full(sparse(beamNo,detectorIds,1));
figure; set(gcf,'pos',[2715   148    1480    1001])
for ii = 1:2:size(beamdet_accept,1)
    plot(beamdet_accept(ii,:),'LineWidth',2); hold on
end
% legend({'beam 1','beam 5','beam 9','beam 13','beam 17'})
legend({'beam 1','beam 3','beam 5','beam 7'})
xlabel('Detector ID')
ylabel('Detected photon counts')
title('Detected photon counts')
set(gca,'FontSize',20)
saveas(gcf,fullfile(dosematrixFolder,['Original_counts.png']))

thresh = median(beamdet_accept(ii,:))*4;
beamNo_detectorIDs_rej = find(beamdet_accept>thresh);
beamNo_detectorIDs = (detectorIds-1)*size(beamdet_accept,1) + beamNo;
badID = ismember(beamNo_detectorIDs,beamNo_detectorIDs_rej);
badIDmask = zeros(size(beamNo_detectorIDs));
badIDmask(badID) = 1;

newbeamdet_accept = full(sparse(beamNo(badIDmask==0),detectorIds(badIDmask==0),1));
figure; set(gcf,'pos',[2715   148    1480    1001])
for ii = 1:4:size(newbeamdet_accept,1)
    plot(newbeamdet_accept(ii,:),'LineWidth',2); hold on
end
legend({'beam 1','beam 5','beam 9','beam 13','beam 17'})
% legend({'beam 1','beam 3','beam 5','beam 7'})
xlabel('Detector ID')
ylabel('Detected photon counts')
title('Detected photon counts')
set(gca,'FontSize',20)
saveas(gcf,fullfile(dosematrixFolder,['Processed_counts.png']))

beamletIDs(badID,:) = [];
detectorIds(badID,:) = [];
beamNo(badID,:) = [];
beamletNo(badID,:) = [];
energy(badID,:) = [];
eventIds(badID,:) = [];
globalTimes(badID,:) = [];
% Mega(badID,:) = [];
save(fullfile(dosematrixFolder,[patientName projectName '_ringdetection.mat']),...
    'detectorIds','beamNo','beamletNo','energy','eventIds','globalTimes','beamletIDs','numevent','-v7.3');

theta_buff = detectorIds/max(detectorIds)*2*pi;
theta_buff0 = floor(mean(beamNo_detectorIDs_rej))/max(detectorIds)*2*pi;
theta = pi - abs(mod(theta_buff - theta_buff0, 2*pi)-pi);

%%
Nbins = 50;
figure;histogram(energy,Nbins)
xlabel('Energy (MeV)')
figure;histogram(energy(energy<1),Nbins)
xlabel('Energy (MeV)')
figure;histogram(energy(energy<0.8&energy>0.2),Nbins)
xlabel('Energy (MeV)')
figure;histogram(energy(energy<0.7&energy>0.3),Nbins)

Nbins = 30;
figure;histogram(theta,Nbins)
figure;histogram(theta(energy<1),Nbins)

X = [energy(energy<1),theta(energy<1)];
[N,c] = hist3(X,'Ctrs',{0:1/Nbins:1 0:pi/Nbins:pi});
figure;heatmap(c{2},c{1},N)

figure;hist3(X,'Ctrs',{0:1/Nbins:1 0:pi/Nbins:pi})
xlabel('energy')
ylabel('theta')

energy_90 = energy(abs(theta-pi/2)<0.001);
save(fullfile(dosematrixFolder,'photonstats.mat'),'energy','energy_90')
Nbins = 50;
figure;histogram(energy_90,Nbins)
xlabel('Energy (MeV)')


%%

Ind_511 = find(abs(energy-0.511)<0.511*0.0001);
Ind_10 = find(abs(energy-0.511)<0.511*0.1);
numel(Ind_511)/numel(Ind_10)
Ind_20 = find(abs(energy-0.511)<0.511*0.2);
numel(Ind_511)/numel(Ind_20)
numel(Ind_511)/numel(energy)

numdetectors = max(detectorIds);

%% Time correction: Adding time of previous events
for doserate = 10/60
    for detectorefficiency = 1
        time = max(dose_onebeam(:))*numevent/doserate;
        
        area = 2*pi*120*10; % cm^2
        flux = numel(energy)/area/time;
        flux_20 = numel(Ind_20)/area/time;
        flux_10 = numel(Ind_10)/area/time;
        flux_511 = numel(Ind_511)/area/time;
        
        fluencerate = numel(energy)/time;
        fluencerate_20 = numel(Ind_20)/time;
        fluencerate_10 = numel(Ind_10)/time;
        fluencerate_511 = numel(Ind_511)/time;
       
        fluencerate_perdetector = numel(energy)/numdetectors/time;
        fluencerate_perdetector_20 = numel(Ind_20)/numdetectors/time;
        fluencerate_perdetector_10 = numel(Ind_10)/numdetectors/time;
        fluencerate_perdetector_511 = numel(Ind_511)/numdetectors/time;
      
     end
end

fluxall = [flux;flux_20;flux_10;flux_511];
fluencerateall = [fluencerate;fluencerate_20;fluencerate_10;fluencerate_511];
fluencerate_perdetectorall = [fluencerate_perdetector;fluencerate_perdetector_20;fluencerate_perdetector_10;fluencerate_perdetector_511];
tableflux = [fluxall fluencerateall fluencerate_perdetectorall];


%%

info = struct();
count = 1;

for doserate = [10/60 1/60 0.1/60]
    for detectorefficiency = [1 0.1]
        info(count).doserate = doserate*6000;        
        info(count).detectorefficiency = detectorefficiency;
        
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
        
        TimeResolution = 0.02; % 400 ps
        CoincidenceTime = 1;
        Ind_coin_511 = IdentifyLOR_511(energy, CorrectedTime, CoincidenceTime);
        
        CorrectedTime_TR = CorrectedTime + TimeResolution*randn(size(CorrectedTime));
        Ind_coin_10 = IdentifyLOR(energy, CorrectedTime_TR, CoincidenceTime, 0.1);
        info(count).TruePositive_10 = length(intersect(Ind_coin_511(:,1).*Ind_coin_511(:,2),Ind_coin_10(:,1).*Ind_coin_10(:,2)))/length(Ind_coin_10(:,1));
        Ind_coin_20 = IdentifyLOR(energy, CorrectedTime_TR, CoincidenceTime, 0.2);
        info(count).TruePositive_20 = length(intersect(Ind_coin_511(:,1).*Ind_coin_511(:,2),Ind_coin_20(:,1).*Ind_coin_20(:,2)))/length(Ind_coin_20(:,1));
        Ind_coin_all = IdentifyLOR(energy, CorrectedTime_TR, CoincidenceTime, 100);
        info(count).TruePositive_all = length(intersect(Ind_coin_511(:,1).*Ind_coin_511(:,2),Ind_coin_all(:,1).*Ind_coin_all(:,2)))/length(Ind_coin_all(:,1));
        
        area = 2*pi*120*10; % cm^2
        info(count).flux_coincident_511 = numel(Ind_coin_511)*2/area/time;
        
        count = count+1;
    end
end

T = struct2table(info);
filename = fullfile(dosematrixFolder,'stats.csv');
delete(filename)
writetable(T,filename)
