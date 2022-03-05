# description: {description:s}
#----------------------------------------------------
# Verbose level 
/control/verbose 1
/process/verbose 0
/run/verbose 1
/event/verbose 0
/geometry/navigator/verbose 0
/tracking/verbose 0

# set number of threads
/run/numberOfThreads {nthreads:d}

#Following Geometry parameters should be set prior run initialization: toggle attenuator, attenuator thickness, detector position
/run/initialize

# adjust global magnetic field
/globalField/setValue {magx:f} {magy:f} {magz:f} {magu:s}
/globalField/verbose 1

# generate HepRap file according to settings in vis.mac
# /control/execute vis.mac
