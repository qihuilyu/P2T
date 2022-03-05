load('C:\Users\admin\Downloads\Detection_doserate_450000.mat')
difftime = diff(sort(signals.time));

Nbins = 100;
figure;histogram(difftime,Nbins)
xlabel('Time (ns)')

figure;histogram(difftime(difftime<1),Nbins)
xlabel('Time (ns)')

figure;histogram(difftime(difftime<0.1),Nbins)
xlabel('Time (ns)')

nnz(find(signals.detector==1))
