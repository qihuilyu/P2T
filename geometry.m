 nb_cryst = 360;
 nb_rings = 1;
 ring_R1 = 120;
 ring_R2 = 125;
 cryst_dX = 5;
   
 dPhi = 2*pi/nb_cryst;
 half_dPhi = 0.5*dPhi;
 cosdPhi = cos(half_dPhi);
 tandPhi = tan(half_dPhi);

 cryst_dY = ring_R1*tandPhi*2;    
 cryst_dZ = ring_R2*cosdPhi-ring_R1;