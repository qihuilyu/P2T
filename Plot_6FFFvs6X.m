load('E:\PPTI\varian phase space data\TrueBeam_v2_6FFF_00.mat','type','energies');
Nbins = 100;
figure;energyhist_6FFF = histogram(energies,Nbins);

load('E:\PPTI\varian phase space data\TrueBeam_v2_6X_00.mat','type','energies');
Nbins = 100;
figure;energyhist_6X = histogram(energies,Nbins);

energysam_6FFF = (energyhist_6FFF.BinEdges(1:end-1) + energyhist_6FFF.BinEdges(2:end))/2;
energyportion_6FFF = energyhist_6FFF.Values/sum(energyhist_6FFF.Values);

energysam_6X = (energyhist_6X.BinEdges(1:end-1) + energyhist_6X.BinEdges(2:end))/2;
energyportion_6X = energyhist_6X.Values/sum(energyhist_6X.Values);

figure;plot(energysam_6FFF,energyportion_6FFF); hold on;
plot(energysam_6X,energyportion_6X); 

legend('flattening filter free (FFF)','with flattening filter')

title('6MV spectrum')
xlabel('Energy (MeV)')

[energysam_6X',energyportion_6X']

