import math
import numpy as np

def inv_rotbeam(vec, angle_gantry, angle_couch, angle_coll):
    """implement transform from Beams-eye-view coordsys to scanner coordsys using the C-arm linac
    rotational degrees of freedom"""
    tmp = rot_around_axis_rhs(vec, [0,1,0], -(angle_couch+angle_coll)) # coll rot + correction
    rot_axis = [math.sin(-angle_couch), 0, math.cos(-angle_couch)]     # couch rotation
    return rot_around_axis_rhs(tmp, rot_axis, angle_gantry)            # gantry rotation

def rot_around_axis_rhs(pp, rr, angle):
    """Rotate point pp around axis rr (already normalized) by angle"""
    cc, ss = math.cos(angle), math.sin(angle)
    return np.array([
        (-rr[0]*(-rr[0]*pp[0] - rr[1]*pp[1] - rr[2]*pp[2]))*(1-cc) + pp[0]*cc + (-rr[2]*pp[1] + rr[1]*pp[2])*ss,
        (-rr[1]*(-rr[0]*pp[0] - rr[1]*pp[1] - rr[2]*pp[2]))*(1-cc) + pp[1]*cc + ( rr[2]*pp[0] - rr[0]*pp[2])*ss,
        (-rr[2]*(-rr[0]*pp[0] - rr[1]*pp[1] - rr[2]*pp[2]))*(1-cc) + pp[2]*cc + (-rr[1]*pp[0] + rr[0]*pp[1])*ss,
    ])
