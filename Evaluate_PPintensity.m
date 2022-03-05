Anni3D = reshape(full(sum(M_Anni,2)),size(masks{1}.mask));
Anni2D = Anni3D(:,:,slicenum);
numevents = max(eventIds);
figure;imshow(Anni2D,[])

% Anni2D_norm = Anni2D*numevents;
% figure;imshow(Anni2D_norm,[1337,2130])
Anni2D_norm = Anni2D_corrected;
figure;imshow(Anni2D_norm,[1337,2130]*0.08)

ROIInd = [[99.1875 82.5625 8.00000000000001 7.87499999999999]
        [102.5 63.5 5 5]
        [89.5 46.5 5 5]
        [68.5 39.5 5 5]
        [48.5 46.5 4 5]
        [35.5 62.5 4 7]
        [35.5 84.5 4 7]
        [48.0625 102.3125 4.125 6.25]
        [68.5 108.5 4 6] 
    ];
ROIrow1 = ceil(ROIInd(:,2));
ROIcolumn1 = ceil(ROIInd(:,1));
ROIrow2 = floor(ROIInd(:,2))+floor(ROIInd(:,4));
ROIcolumn2 = floor(ROIInd(:,1))+floor(ROIInd(:,3));

for i = 1
    for j = 1:size(ROIInd,1)
        ImgROI = Anni2D_norm(ROIrow1(j):ROIrow2(j),ROIcolumn1(j):ROIcolumn2(j),i);
        VF(j) = mean(ImgROI(:));
    end
end

test = (VF-VF(1))/VF(1)*100;
figure;scatter([53,56,64,70,73,79,83],test(3:end))

xlabel('Atomic No')
ylabel('Increased contrast to water (%)')
set(gca,'FontSize',15)
refline



