load('D:\datatest\PairProd\PartialViewPhantom_LimitedROI_2mmbeamlet_230m_merged\PairProd\results\Recon_pairprod.mat')

i = 1;
ImgInfo(i).Img_raw = Anni2D.*mask0;
ImgInfo(i).Img_corrected = Anni2D_corrected.*mask0;
ImgInfo(i).ImgNorMax = ImgInfo(i).Img_raw/max(ImgInfo(i).Img_raw(:));
ImgInfo(i).Method = 'PPI ground-truth';
i = i+1;
ImgInfo(i).Img_raw = img_fbp.*mask0;
ImgInfo(i).Img_corrected = img_fbp_corrected.*mask0;
ImgInfo(i).ImgNorMax = ImgInfo(i).Img_raw/max(ImgInfo(i).Img_raw(:));
ImgInfo(i).Method = 'PPI FBP';
i = i+1;
ImgInfo(i).Img_raw = img_beampath.*mask0;
ImgInfo(i).Img_corrected = img_beampath_corrected.*mask0;
ImgInfo(i).ImgNorMax = ImgInfo(i).Img_raw/max(ImgInfo(i).Img_raw(:));
ImgInfo(i).Method = 'PPI beam-path';
i = i+1;
ImgInfo(i).Img_raw = img_direct.*mask0;
ImgInfo(i).Img_corrected = img_direct_corrected.*mask0;
ImgInfo(i).ImgNorMax = ImgInfo(i).Img_raw/max(ImgInfo(i).Img_raw(:));
ImgInfo(i).Method = 'PPI reconstruction-less';

figure;imshow([ImgInfo.Img_corrected],[])


%%

ROIInd = [
        [53.5 60.5 5 5]  % 8
        [65.5 77.5 5 5]  % 9
        [86.5 84.5 5 5]  % 10
        [99.5 87.5 13 9]
    ]; 

ROIrow1 = ceil(ROIInd(:,2));
ROIcolumn1 = ceil(ROIInd(:,1));
ROIrow2 = floor(ROIInd(:,2))+floor(ROIInd(:,4));
ROIcolumn2 = floor(ROIInd(:,1))+floor(ROIInd(:,3));

for i = 1:numel(ImgInfo)
    Img = ImgInfo(i).Img_corrected;
%     ind1 = -25; ind2 = 18;
%     Img = TranslateFigure(Img,-ind1,-ind2);

    j = size(ROIInd,1);
    ImgROI = Img(ROIrow1(j):ROIrow2(j),ROIcolumn1(j):ROIcolumn2(j));
    imginten0 = mean(ImgROI(:))
    
    for j = 1:size(ROIInd,1)
        ImgROI = Img(ROIrow1(j):ROIrow2(j),ROIcolumn1(j):ROIcolumn2(j));
        ImgROI = ImgROI/imginten0;
        imginten(i,j) = mean(ImgROI(:));
        imgnoise(i,j) = std(ImgROI(:));
        ImgInfo(i).ImgNor = ImgInfo(i).Img_corrected/imginten0;
    end
end

figure;imshow([ImgInfo.ImgNor],[0.2 1.7])


%%
waterind = 4;
Contrast = (imginten-imginten(:,waterind))./repmat(imginten(:,waterind),[1,size(imginten,2)])*100;
AtomicNum = [73,79,83];
MarkerSize = 50;
LineWidth = 2;
% yyaxis right;
% scatter(AtomicNum,test(4,3:end),MarkerSize,'s','filled');  hold on;
% ylabel('Increased contrast to water (%), CT')
% legend({'CT'})
NumSamples = size(Contrast,2)-1;
xsample = 1:3;
selectedmethod = [1,2,3,4];
figure; 
for method = selectedmethod
    scatter(AtomicNum,Contrast(method,xsample),MarkerSize,'o','filled'); hold on;
end
refline;
ylabel('Increased contrast to water (%)')
legend({ImgInfo(selectedmethod).Method})
xlabel('Atomic No')
set(gca,'FontSize',15)
saveas(gcf,'D:\datatest\PairProd\GoodResult\nanoparticle_limitedFOV_linearrelationship.png');
saveas(gcf,'D:\datatest\PairProd\GoodResult\nanoparticle_limitedFOV_linearrelationship.pdf');




% scatter(xsample,Contrast(1,xsample),MarkerSize,'o','filled'); hold on;
% scatter(xsample,Contrast(2,xsample),MarkerSize,'+','LineWidth',LineWidth);  hold on;
% scatter(xsample,Contrast(3,xsample),MarkerSize,'d','LineWidth',LineWidth);  hold on;
% scatter(xsample,Contrast(4,xsample),MarkerSize,'.','LineWidth',LineWidth);  hold on;
% scatter(xsample,Contrast(5,xsample),MarkerSize,'*','LineWidth',LineWidth);  hold on;


% icolor = 'r';
% figure;scatter(AtomicNum,test(1,3:end),[],icolor);hline=refline; hline.Color = icolor;  hold on;
% icolor = 'y';
% scatter(AtomicNum,test(2,3:end),[],icolor);hline2=refline; hline2.Color = icolor;
% icolor = 'b';
% scatter(AtomicNum,test(3,3:end),[],icolor);hline3=refline; hline3.Color = icolor;  hold on;

% 
% 
% figure; 
% scatter(AtomicNum,Contrast(4,3:end),MarkerSize,'s','filled');  hold on;
% ylabel('Increased contrast to water (%)')
% legend({'CT'})
% xlabel('Atomic No')
% set(gca,'FontSize',15)


%%


figure;imagesc(PP_Dose,[0,0.033]); colorbar; colormap(jet)
axis off
axis equal
saveas(gcf, 'D:\datatest\PairProd\GoodResult\nanoparticleLimitedFOV_Dose_all.png')



