C = 6;
H = 1;
N = 7;
O = 8;
Na = 11;
P = 30;
S = 16;
Cl = 17;
K = 19;
Mg = 12;
Ca = 20;
Fe = 26;

elN=0.7;
elO=0.3;
Air = elN*N + elO*O;
Air_CT = elN*N^3 + elO*O^3;
Air_dens = 1.290e-03;

elH=0.103;
elC=0.105;
elN=0.031;
elO=0.749;
elNa=0.002;
elP=0.002;
elS=0.003;
elCl=0.002;
elK=0.003;
lunginhale = elH*H + elC*C + elN*N + elO*O + elNa*Na + elP*P + elS*S + elCl*Cl + elK*K;
lunginhale_CT = elH*H^3 + elC*C^3 + elN*N^3 + elO*O^3 + elNa*Na^3 + elP*P^3 + elS*S^3 + elCl*Cl^3 + elK*K^3;
lunginhale_dens = 0.217;

elH=0.103;
elC=0.105;
elN=0.031;
elO=0.749;
elNa=0.002;
elP=0.002;
elS=0.003;
elCl=0.002;
elK=0.003;
lungexhale = elH*H + elC*C + elN*N + elO*O + elNa*Na + elP*P + elS*S + elCl*Cl + elK*K;
lungexhale_CT = elH*H^3 + elC*C^3 + elN*N^3 + elO*O^3 + elNa*Na^3 + elP*P^3 + elS*S^3 + elCl*Cl^3 + elK*K^3;
lungexhale_dens = 0.508;

elH=0.114;
elC=0.598;
elN=0.007;
elO=0.278;
elNa=0.001;
elP=0;
elS=0.001;
elCl=0.001;
elK=0;
Adipose = elH*H + elC*C + elN*N + elO*O + elNa*Na + elP*P + elS*S + elCl*Cl + elK*K;
Adipose_CT = elH*H^3 + elC*C^3 + elN*N^3 + elO*O^3 + elNa*Na^3 + elP*P^3 + elS*S^3 + elCl*Cl^3 + elK*K^3;
Adipose_dens = 0.967;

elH=0.109;
elC=0.506;
elN=0.023;
elO=0.358;
elNa=0.001;
elP=0.001;
elS=0.001;
elCl=0.001;
elK=0;
Breast = elH*H + elC*C + elN*N + elO*O + elNa*Na + elP*P + elS*S + elCl*Cl + elK*K;
Breast_CT = elH*H^3 + elC*C^3 + elN*N^3 + elO*O^3 + elNa*Na^3 + elP*P^3 + elS*S^3 + elCl*Cl^3 + elK*K^3;
Breast_dens = 0.990;

elH=0.112;
elO=0.888;
Water = elH*H + elO*O;
Water_CT = elH*H^3 + elO*O^3;
Water_dens = 1;

elH=0.102;
elC=0.143;
elN=0.034;
elO=0.710;
elNa=0.001;
elP=0.002;
elS=0.003;
elCl=0.001;
elK=0.004;
muscle = elH*H + elC*C + elN*N + elO*O + elNa*Na + elP*P + elS*S + elCl*Cl + elK*K;
muscle_CT = elH*H^3 + elC*C^3 + elN*N^3 + elO*O^3 + elNa*Na^3 + elP*P^3 + elS*S^3 + elCl*Cl^3 + elK*K^3;
muscle_dens = 1.061;

elH=0.102;
elC=0.139;
elN=0.030;
elO=0.716;
elNa=0.002;
elP=0.003;
elS=0.003;
elCl=0.002;
elK=0.003;
liver = elH*H + elC*C + elN*N + elO*O + elNa*Na + elP*P + elS*S + elCl*Cl + elK*K;
liver_CT = elH*H^3 + elC*C^3 + elN*N^3 + elO*O^3 + elNa*Na^3 + elP*P^3 + elS*S^3 + elCl*Cl^3 + elK*K^3;
liver_dens = 1.071;

elH=0.085;
elC=0.404;
elN=0.058;
elO=0.367;
elNa=0.001;
elP=0.034;
elS=0.002;
elCl=0.002;
elK=0.001;
elMg=0.001;
elCa=0.044;
elFe=0.001;
trabecularBone = elH*H + elC*C + elN*N + elO*O + elNa*Na + elP*P + elS*S + elCl*Cl + elK*K + elMg*Mg + elCa*Ca + elFe*Fe;
trabecularBone_CT = elH*H^3 + elC*C^3 + elN*N^3 + elO*O^3 + elNa*Na^3 + elP*P^3 + elS*S^3 + elCl*Cl^3 + elK*K^3 + elMg*Mg + elCa*Ca + elFe*Fe;
trabecularBone_dens = 1.159;

elH=0.056;
elC=0.235;
elN=0.050;
elO=0.434;
elNa=0.001;
elMg=0.001;
elP=0.072;
elS=0.003;
elCl=0.001;
elK=0.001;
elCa=0.146;
elFe=0;
denseBone = elH*H + elC*C + elN*N + elO*O + elNa*Na + elP*P + elS*S + elCl*Cl + elK*K + elMg*Mg + elCa*Ca + elFe*Fe;
denseBone_CT = elH*H^3 + elC*C^3 + elN*N^3 + elO*O^3 + elNa*Na^3 + elP*P^3 + elS*S^3 + elCl*Cl^3 + elK*K^3 + elMg*Mg + elCa*Ca + elFe*Fe;
denseBone_dens = 1.575;

Air*Air_dens/Water-1
lunginhale*lunginhale_dens/Water-1
lungexhale*lungexhale_dens/Water-1
Adipose*Adipose_dens/Water-1
Breast*Breast_dens/Water-1
muscle*muscle_dens/Water-1
liver*liver_dens/Water-1
trabecularBone*trabecularBone_dens/Water-1
denseBone*denseBone_dens/Water-1






