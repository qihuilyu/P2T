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

# define beam spectrum (10MV)
/gps/ene/type User
/gps/hist/type energy
/gps/hist/point 0.15 0.0239
/gps/hist/point 0.20 0.0505
/gps/hist/point 0.30 0.0659
/gps/hist/point 0.40 0.0489
/gps/hist/point 0.50 0.0620
/gps/hist/point 0.60 0.0673
/gps/hist/point 0.80 0.0814
/gps/hist/point 1.00 0.0742
/gps/hist/point 1.25 0.0622
/gps/hist/point 1.50 0.0823
/gps/hist/point 2.00 0.1160
/gps/hist/point 3.00 0.0966
/gps/hist/point 4.00 0.0597
/gps/hist/point 5.00 0.0405
/gps/hist/point 6.00 0.0376
/gps/hist/point 8.00 0.0254
/gps/hist/point 10.00 0.0055



