# description: {description:s}
#----------------------------------------------------
# define General Particle Source with 5x5mm field size at origin directed in +y
/gps/particle e-
/gps/pos/type Plane
/gps/pos/shape Circle
/gps/pos/centre {cx:f} {cy:f} {cz:f} {cu:s}
/gps/pos/rot1 {rot1x:f} {rot1y:f} {rot1z:f}
/gps/pos/rot2 {rot2x:f} {rot2y:f} {rot2z:f}
# field size at origin will be 20*half(x,y)
/gps/pos/halfx {fsx:f} {fsu:s}
/gps/pos/halfy {fsy:f} {fsu:s}
/gps/ang/type focused
/gps/ang/focuspoint {fx:f} {fy:f} {fz:f} {fu:s}

# define beam energy
/gps/ene/type Mono
/gps/ene/mono {energy:f} MeV
