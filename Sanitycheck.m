% Anni3D = reshape(full(sum(M_Anni,2)),size(masks{1}.mask));
% Anni2D = Anni3D(:,:,ceil(end/2));
% sz = size(Anni2D);
% 
% x_norm = Anni2D/sum(Anni2D(:));

clear
close all
clc

R1 = 1200;
distrange = 500;
imgsize = [151,153];
nb_cryst = 1440;
imgres = 3;

patientName = 'phantom_10cm';
projectName = 'PairProd';
patFolder = fullfile('D:\datatest\PairProd\',patientName);
projectFolder = fullfile(patFolder,projectName);
dosematrixFolder = fullfile(projectFolder,'dosematrix');
load(fullfile(dosematrixFolder,[patientName projectName '_dicomimg.mat']),'img','imgres');

x_norm = img(:,:,end/2);

% x_norm = zeros(imgsize);
% x_norm(35:120,33:110) = 0.5;
% x_norm(73:77, 75:79) = 1;
% x_norm(123:127, 55:59) = 1;
% x_norm(33:37, 85:89) = 1;
% x_norm(43:47, 35:39) = 1;
% x_norm(133:137, 105:109) = 1;

x_norm = x_norm/sum(x_norm(:));
img = x_norm;
x_cumsum = [0;cumsum(x_norm(:))];

Nsample = 1e+06;
x_rand = rand(Nsample,1);
ang_rand = rand(Nsample,1)*pi;

Nbatch = 1e+05; minI = zeros(Nsample,1);
for ii = 1:Nsample/Nbatch
    ind = (ii-1)*Nbatch+1 : ii*Nbatch;
    minIs = sum(x_cumsum-x_rand(ind)'<0,1);
    minI(ind) = minIs(:);
end
[x0_sub,y0_sub] = ind2sub(imgsize,minI);

xp0_loc = (x0_sub - (1+imgsize(1))/2)*imgres;
yp0_loc = (y0_sub - (1+imgsize(2))/2)*imgres;

arcsin = asin(cos(ang_rand).*yp0_loc/R1 - sin(ang_rand).*xp0_loc/R1);
alpha1 = mod(arcsin + ang_rand,2*pi);
alpha2 = mod(pi + ang_rand - arcsin,2*pi);

xp1_loc = R1*cos(alpha1);
yp1_loc = R1*sin(alpha1);
xp2_loc = R1*cos(alpha2);
yp2_loc = R1*sin(alpha2);
r1 = sqrt((xp1_loc-xp0_loc).^2 + (yp1_loc-yp0_loc).^2);
r2 = sqrt((xp2_loc-xp0_loc).^2 + (yp2_loc-yp0_loc).^2);
clight = 300; % c = 300mm/ns
deltat = (r1-r2)/clight;

% figure;scatter(xp0_loc,yp0_loc)

%% Sanity check
% x = [xp0_loc xp1_loc xp2_loc];
% y = [yp0_loc yp1_loc yp2_loc];
% 
% figure;axis equal;
% for ii = 10000:10010
%     scatter(x(ii,:),y(ii,:));hold on;
% end

%% Direct iamge generation
img_gt = zeros(imgsize(1),imgsize(2));
xImage = ((1:imgsize(1))-(1+imgsize(1))/2)*imgres;
yImage = ((1:imgsize(2))-(1+imgsize(2))/2)*imgres;

for ii = 1:length(xp0_loc)
    ixp0 = xp0_loc(ii);
    iyp0 = yp0_loc(ii);
    
    if(ixp0<min(xImage)||ixp0>max(xImage)||iyp0<min(yImage)||iyp0>max(yImage))
         continue
    end
    
    [~,xind] = min(abs(ixp0-xImage));
    [~,yind] = min(abs(iyp0-yImage));

    img_gt(xind,yind) = img_gt(xind,yind) + 1;
end
figure;imshow(img_gt,[])

%% FBP
detid_pair = mod(round([alpha1 alpha2]/(2*pi)*nb_cryst),nb_cryst) + 1;
[sino, dr, newunidist, sinobuff, unidist] = rebin_PET2(detid_pair, nb_cryst, R1, distrange);
ig = image_geom('nx', size(img,2), 'ny', size(img,1), 'fov', size(img,1)*imgres);
sg = sino_geom('par', 'nb', size(sino,1), 'na', size(sino,2), 'dr', dr);
img_fbp = em_fbp_QL(sg, ig, sino)';

G = Gtomo2_strip(sg, ig);
ForBack.applyFP = @(x) G*x(:);
sino_FP = reshape(ForBack.applyFP(x_norm'),size(sino));
img_fbp_gtsino = em_fbp_QL(sg, ig, sino_FP)';

figure;imshow([sino/sum(sino(:)); sino_FP/sum(sino_FP(:))],[])
figure;imshow([sinobuff/sum(sinobuff(:)); sino/sum(sino(:)); sino_FP/sum(sino_FP(:))],[])
figure;imshow([img_fbp_gtsino/max(img_fbp_gtsino(:)) img_fbp/max(img_fbp(:))],[])
figure;imshow([sino/sum(sino(:))-sino_FP/sum(sino_FP(:))],[])

%% Reconstruction-less image generation
reconparams = struct('nb_cryst',nb_cryst,'R1',R1,'distrange',distrange,...
    'imgres',imgres,'imgsize',[ig.nx, ig.ny]);
img_direct = recon_TOF_direct(reconparams, detid_pair, deltat)';
figure;imshow([x_norm/max(x_norm(:))  img_gt/max(img_gt(:)) img_fbp/max(img_fbp(:)) img_direct/max(img_direct(:))],[])

%% TOF reconstruction
img_fbp = img_fbp/mean(mean(img_fbp(108:113,75:80)));
x_norm = x_norm/mean(mean(x_norm(108:113,75:80)));
img_gt = img_gt/mean(mean(img_gt(108:113,75:80)));
img_direct = img_direct/mean(mean(img_direct(108:113,75:80)));

count = 1;
resultFolder = 'D:\datatest\PairProd\sanitycheck';
for TR = [0.002,0.02,0.2,0.6,1.2,2]
    [sino3D, dr, sinobuff3D] = rebin_PET_TOF(reconparams, detid_pair, deltat, TR);
    sino3DAll{count} = sino3D;
    sinobuff3DAll{count} = sinobuff3D;
    sino3DAll{count} = flip(sino3D,2);
    img_toffbp{count} = em_toffbp_QL(sg, ig, sino3DAll{count})';
    img_tofbp{count} = em_tof_backprojection_QL(sg, ig, sino3DAll{count})';
   
    img_toffbp{count} = img_toffbp{count}/mean(mean(img_toffbp{count}(108:113,75:80)));
    figure;imshow([x_norm img_fbp img_toffbp{count} img_direct],[0,1])
%     figure;imshow([x_norm/max(x_norm(:)) img_fbp/max(img_fbp(:)) img_toffbp{count}/max(img_toffbp{count}(:)) img_direct/max(img_direct(:))],[0,1])
    title(['TR: ' num2str(TR) 's'])
    saveas(gcf,fullfile(resultFolder,['imgrecon_TR' num2str(TR) '_simulated_phantom.png']))
    
    count = count+1;
end
