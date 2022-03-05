clear
close all
clc

patientName = 'GBMHY_final_100m_run01torun10_1000MUpermin';
projectName = 'PairProd';
patFolder = fullfile('D:\datatest\PairProd\',patientName);
OutputFileName = fullfile('D:\datatest\PairProd\','GBMHY.mat');
CERR('CERRSLICEVIEWER')
sliceCallBack_QL('OPENNEWPLANC', OutputFileName);

projectFolder = fullfile(patFolder,projectName);
dosecalcFolder = fullfile(patFolder,'dosecalc');
dosematrixFolder = fullfile(projectFolder,'dosematrix');
resultsFolder = fullfile(projectFolder,'results');
mkdir(resultsFolder)

load('D:\datatest\PairProd\GBMHY_final_100m_run01torun10_1000MUpermin\PairProd\results\Recon_pairprod_direct_sigma0_2.mat')
load('D:\datatest\PairProd\GBMHY_final_100m_run01torun10_1000MUpermin\PairProd\results\Recon_pairprod_fbp.mat')
load('D:\datatest\PairProd\GBMHY_final_100m_run01torun10_1000MUpermin\PairProd\results\Dose_TERMA_PPTI.mat')
% load(fullfile(dosematrixFolder,[patientName projectName '_dicomimg.mat']),'img','imgres');
load('D:\datatest\PairProd\GBMHY_final_100m\PairProd\params\StructureInfo7.mat')

paramsFolder = fullfile(projectFolder,'params');
ParamsNum = 0;
load(fullfile(paramsFolder,['params' num2str(ParamsNum) '.mat']),'params');


%%
BODY = (StructureInfo(1).Mask | StructureInfo(2).Mask);
PTV = StructureInfo(1).Mask;

xoff = -0.2;
yoff = 0.2;

dose3D(dose3D<0) = 0;
dose3D = dose3D/mean(dose3D(PTV==1));
dose3D(BODY==0) = 0;
planName = [patientName 'dose3D'];
addDoseToGui_Move_QL(dose3D,[planName],xoff,yoff)

TERMA3D(TERMA3D<0) = 0;
TERMA3D = TERMA3D/mean(TERMA3D(PTV==1));
TERMA3D(BODY==0) = 0;
planName = [patientName 'TERMA3D'];
addDoseToGui_Move_QL(TERMA3D,[planName],xoff,yoff)

Anni3D(Anni3D<0) = 0;
Anni3D = Anni3D/mean(Anni3D(PTV==1));
Anni3D(BODY==0) = 0;
planName = [patientName 'Anni3D'];
addDoseToGui_Move_QL(Anni3D,[planName],xoff,yoff)




%%
i = 1;
DoseInfo(i).Name = 'dose';
DoseInfo(i).Data = dose3D;
DoseInfo(i).Date = datestr(datetime);
i = i+1;
DoseInfo(i).Name = 'TERMA';
DoseInfo(i).Data = TERMA3D;
DoseInfo(i).Date = datestr(datetime);
i = i+1;
DoseInfo(i).Name = 'PPTI';
DoseInfo(i).Data = Anni3D;
DoseInfo(i).Date = datestr(datetime);
save(fullfile(patFolder,[patientName '_DoseInfo.mat']),'DoseInfo','-v7.3');

strNum = [1,3,7,9];
numBins = 200;
scale = plotDVH_QL(DoseInfo([1:3]), strNum, StructureInfo, numBins, 0);

set(gcf,'pos',[0.1109    0.2882    0.7211    0.5625])
LegendSize = 15;
LabelSize = 25;
LineWidth = 1.5;
AxesSize = 15;
TitleSize = 15;
LegendLocation = 'bestoutside';
FigureColor = 'white';
set(gcf,'color',FigureColor);
title('')
xlabel('Normalized intensity');
h=legend({'PTV','R Optic Nerve','Brainstem','Ring structure','PTV','R Optic Nerve','Brainstem','Ring structure','PTV','R Optic Nerve','Brainstem','Ring structure'});
    


%%
set(gcf,'PaperPositionMode','auto')
fig=gcf;                                     % your figure
fig.PaperPositionMode='auto';
saveas(gcf,fullfile(dvhfolder,'DVH_all'),'svg');
export_fig(fullfile(dvhfolder,'DVH_all.pdf'))
% print(gcf, fullfile(dvhfolder,'DVH_all'),'-dtiff','-r600');



%%
slicenum = 88;
ind1 = 20; ind2 = 20;
BODY2D = TranslateFigure(BODY(:,:,slicenum),ind1,ind2);        
PTV2D = TranslateFigure(PTV(:,:,slicenum),ind1,ind2);        

xoff = -5;
yoff = 5.4;

img_direct(img_direct<0) = 0;
img_direct = img_direct/mean(img_direct(PTV2D==1));
img_direct(BODY2D==0) = 0;
img_direct3D = repmat(img_direct,[1,1,size(Anni3D,3)]);
planName = [patientName 'img_direct'];
addDoseToGui_Move_QL(img_direct3D,[planName],xoff,yoff)

img_fbp(img_fbp<0) = 0;
img_fbp = img_fbp/mean(img_fbp(PTV2D==1));
img_fbp(BODY2D==0) = 0;
img_fbp3D = repmat(img_fbp,[1,1,size(Anni3D,3)]);
planName = [patientName 'img_fbp'];
addDoseToGui_Move_QL(img_fbp3D,[planName],xoff,yoff)

%%
load('D:\datatest\patient\patInfo.mat')

% save('D:\datatest\patient\patInfo.mat','patInfo')


%% Dose Wash
patientName = 'GBMHY_PairProd';
ImageSize = [1,5];
jj=find(strcmp({patInfo.Name},patientName));
strNum = patInfo(jj).strNum;
StructureInfo = patInfo(jj).StructureInfo;

FigureNum = get(gcf,'Number');
global planC stateS
patInfo(16).coordInd = [197,197,71];
patInfo(16).colorbarRange = [0,1.1];
ChangeCERRdoseWash_QL(patientName,patInfo)

figuresFolder = ['D:\datatest\PairProd\GoodResult\dosewash\'];
mkdir(figuresFolder)
figureName = [patientName projectName];
[EntireImg,Imgs,ImgsInit,Masks] = SaveDoseWash_QL(patInfo, figuresFolder,figureName,[9:13],patientName,FigureNum,ImageSize);

%%
Imgsselected = {Imgs{1} Imgs{4} Imgs{7} Imgs{10} Imgs{13}};
margin = [70,70];
Imgfinal = PutImgTogether_rectangular(Imgsselected,margin);
figure;imshow(Imgfinal)
set(gca, 'units', 'normalized'); %Just making sure it's normalized
Tight = get(gca, 'TightInset');  %Gives you the bording spacing between plot box and any axis labels                                %[Left Bottom Right Top] spacing
NewPos = [0,0,1,1]; %New plot position [X Y W H]
set(gca, 'Position', NewPos);
saveas(gcf,fullfile(figuresFolder,'dosewash.png'))
saveas(gcf,fullfile(figuresFolder,'dosewash.tiff'))
% 
% %% Difference
% 
% global planC
% Anni2D = TranslateFigure(Anni3D(:,:,slicenum),ind1,ind2);        
% Anni2D(Anni2D<0) = 0;
% Anni2D = Anni2D/mean(Anni2D(PTV2D==1));
% Anni2D(BODY2D==0) = 0;
% img_direct_diff = repmat(img_direct-Anni2D,[1,1,size(Anni3D,3)]);
% planName = [patientName 'img_direct DIFF'];
% addDoseToGui_Move_QL(img_direct_diff,[planName],xoff,yoff)
% img_fbp_diff = repmat(img_fbp-Anni2D,[1,1,size(Anni3D,3)]);
% planName = [patientName 'img_fbp DIFF'];
% addDoseToGui_Move_QL(img_fbp_diff,[planName],xoff,yoff)
% 
% 
% GT = planC{1,9}(10).doseArray;
% planName = [patientName 'img_direct DIFF'];
% addDoseToGui_Move_QL(planC{1,9}(11).doseArray-GT,[planName],xoff,yoff)
% planName = [patientName 'img_fbp DIFF'];
% addDoseToGui_Move_QL(planC{1,9}(12).doseArray-GT,[planName],xoff,yoff)
% 
% GT = planC{1,9}(10).doseArray;
% planName = [patientName 'img_direct DIFF'];
% addDoseToGui_Move_QL(planC{1,9}(11).doseArray-GT,[planName],xoff,yoff)
% planName = [patientName 'img_fbp DIFF'];
% addDoseToGui_Move_QL(planC{1,9}(12).doseArray-GT,[planName],xoff,yoff)
% 
