clear
close all
clc
% 
% t = tic;
% while toc(t)<7200*3
%     pause(2)
% end
% !('/media/raid1/qlyu/PairProd/datatest/collect_doescalc_pairprod.sh')
% 
% patientName = 'PartialViewPhantom_LimitedROI_Allphotons_10MV_10m'; % 
% projectName = 'PairProd';
% patFolder = fullfile('/media/raid1/qlyu/PairProd/datatest',patientName);
% dosecalcFolder = fullfile(patFolder,'dosecalc');
% h5file = fullfile(dosecalcFolder,'PairProd_beamletdose.h5');
% maskfile = fullfile(dosecalcFolder,'PairProd_masks.h5');
% fmapsfile = fullfile(dosecalcFolder,'PairProd_fmaps.h5');
% Anni3Dfile = fullfile(dosecalcFolder,'PairProd_NofPositronAnni3D.h5');
% DetectedEventsfile = fullfile(dosecalcFolder,'PairProd_DetectedEvents.h5');


%% Ring detection
folderName = 'D:\datatest\PairProd\positron_travel_before_annihilation\code\build\run000';
% folderName = '/media/raid1/qlyu/PairProd/experiments/6X/run000';
DetectedEventsfile = fullfile(folderName,'DetectedEvents.h5');
info = h5info(DetectedEventsfile);
detectorIds = double(h5read(DetectedEventsfile, '/detID')); %
energy = h5read(DetectedEventsfile, '/energy'); %
eventIds = double(h5read(DetectedEventsfile, '/EventID')) + 1; %
Tracklength = h5read(DetectedEventsfile, '/Tracklength'); %
% Mega = [globalTimes,eventIds,energy,beamletNo,beamNo,detectorIds];

numevent = max(eventIds);
numevent = numevent + 9999 - mod(numevent-1,10000);

test = [eventIds(2:end) detectorIds(2:end) diff(detectorIds)];
%%
energyid = find(diff(detectorIds)==-1)+1;
sel_energy = energy(energyid);

Tracklengthid = find(detectorIds==1);
sel_Tracklength = Tracklength(Tracklengthid);

Nbins = 30;
figure;subplot(1,2,1);histogram(sel_Tracklength(sel_Tracklength<30),Nbins)
xlabel("positron traveling before annihilation (mm)")
ylabel("count")
set(gca,'FontSize',15)
median(sel_Tracklength)

Nbins = 30;
subplot(1,2,2);histogram(sel_energy,Nbins)
median(sel_energy)
xlabel("initial positron energy (MeV)")
ylabel("count")
set(gca,'FontSize',15)

saveas(gcf,fullfile(folderName,'hist.png'))

