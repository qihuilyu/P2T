clc; clear all; close all; fclose all;
% fname = 'Z:\SharedProjectData\varian_phasespace_data\photon\6X\TrueBeam_v2_6X_00.IAEAphsp';
fname = 'Z:\SharedProjectData\varian_phasespace_data\photon\6FFF\TrueBeam_v2_6FFF_00.IAEAphsp';

% fname = 'E:\PPTI\varian phase space data\example\100.IAEAphsp';
%           // type          1 byte
%           // Energy        4 bytes

%     1     // X is stored ? 4 bytes
%     1     // Y is stored ? 4 bytes
%     0     // Z is stored ?
%     1     // U is stored ? 4 bytes
%     1     // V is stored ? 4 bytes
%     1     // W is stored ?  //W is not stored; set to 1 to enable BEAMnrc import
%     0     // Weight is stored ?
%     0     // Extra floats stored ?
%     0     // Extra longs stored ?

f = dir(fname);
linesize=21;

numparticles=f.bytes/linesize;

fid=fopen(fname,'rb');

for i=1:numparticles
    type{i} = fread(fid,1,'char');
    E(i) = fread(fid,1,'real*4');
    X(i) = fread(fid,1,'real*4');
    Y(i) = fread(fid,1,'real*4');
    U(i) = fread(fid,1,'real*4');
    V(i) = fread(fid,1,'real*4');
end

energies = -E(E<0);
Nbins = 100;
figure;energyhist = histogram(energies,Nbins);

save('E:\PPTI\varian phase space data\TrueBeam_v2_6FFF_00.mat','type','E','X','Y','U','V','energies');

energysam = (energyhist.BinEdges(1:end-1) + energyhist.BinEdges(2:end))/2;
energyportion = energyhist.Values/sum(energyhist.Values);

[energysam' energyportion']

%%


spec6MV = [
0.00 0.0000
0.20 0.0010
0.30 0.0100
0.40 0.0200
0.50 0.0300
0.60 0.0680
0.80 0.0900
1.00 0.1010
1.25 0.1000
1.50 0.1310
2.00 0.1880
3.00 0.1400
4.00 0.0900
5.00 0.0300
6.00 0.0050];


for ii  = 1:size(spec6MV,1)-1
    count(ii) = nnz(find(energies<spec6MV(ii+1,1) & energies>spec6MV(ii,1)));
end


figure;plot(spec6MV(1:end-1,1),count)

figure;plot(spec6MV(1:end-1,1),spec6MV(1:end-1,2))


