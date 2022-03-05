# description: {description:s}
#----------------------------------------------------
# define General Particle Source with 5x5mm field size at origin directed in +y
/gps/particle gamma
/gps/pos/type Plane
/gps/pos/shape Rectangle
/gps/pos/centre {cx:f} {cy:f} {cz:f} {cu:s}
/gps/pos/rot1 {rot1x:f} {rot1y:f} {rot1z:f}
/gps/pos/rot2 {rot2x:f} {rot2y:f} {rot2z:f}
# field size at origin will be 20*half(x,y)
/gps/pos/halfx {fsx:f} {fsu:s}
/gps/pos/halfy {fsy:f} {fsu:s}
/gps/ang/type focused
/gps/ang/focuspoint {fx:f} {fy:f} {fz:f} {fu:s}

# define beam spectrum (6MV)
/gps/ene/type User
/gps/hist/type energy
/gps/hist/point 0.00 0.0000
/gps/hist/point 0.20 0.0010
/gps/hist/point 0.30 0.0100
/gps/hist/point 0.40 0.0200
/gps/hist/point 0.50 0.0300
/gps/hist/point 0.60 0.0680
/gps/hist/point 0.80 0.0900
/gps/hist/point 1.00 0.1010
/gps/hist/point 1.25 0.1000
/gps/hist/point 1.50 0.1310
/gps/hist/point 2.00 0.1880
/gps/hist/point 3.00 0.1400
/gps/hist/point 4.00 0.0900
/gps/hist/point 5.00 0.0300
/gps/hist/point 6.00 0.0050
