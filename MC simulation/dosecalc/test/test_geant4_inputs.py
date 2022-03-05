import unittest
import sys
import os
from os.path import join as pjoin
import math

sys.path.insert(0, os.path.abspath(pjoin(os.pardir, 'webapi')))

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if ('-v' in sys.argv or '--verbose' in sys.argv) else logging.INFO)
logger.addHandler(logging.StreamHandler())

import numpy as np

os.chdir(os.path.abspath(os.path.dirname(__file__)))

todeg = 180.0/math.pi
torad = 1.0/todeg

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def normalize_vect(v):
    return v / np.linalg.norm(v)

class TestMCSource(unittest.TestCase):
    test_cases = [
        ((   0,   1,   0), (   0,   0,   0)),
        ((  -1,   0,   0), (  90,   0,   0)),
        ((   0,  -1,   0), ( 180,   0,   0)),
        ((   1,   0,   0), ( 270,   0,   0)),
        ((   0,   1,   0), ( 360,   0,   0)),
        ((   0,   0,  -1), (  90,   90,  0)),
        ((   0,  -1,   0), ( 180,   90,  0)),
        ((   0,   0,   1), (  90,  -90,  0)),

        ((   0,   1,   0), (   0,   0,  20)),
        ((  -1,   0,   0), (  90,   0,  20)),
        ((   0,  -1,   0), ( 180,   0,  20)),
        ((   1,   0,   0), ( 270,   0,  20)),
        ((   0,   1,   0), ( 360,   0,  20)),
        ((   0,   0,  -1), (  90,   90, 20)),
        ((   0,  -1,   0), ( 180,   90, 20)),
        ((   0,   0,   1), (  90,  -90, 20)),

        ((   0,   1,   0), (   0,   0, -20)),
        ((  -1,   0,   0), (  90,   0, -20)),
        ((   0,  -1,   0), ( 180,   0, -20)),
        ((   1,   0,   0), ( 270,   0, -20)),
        ((   0,   1,   0), ( 360,   0, -20)),
        ((   0,   0,  -1), (  90,   90,-20)),
        ((   0,  -1,   0), ( 180,   90,-20)),
        ((   0,   0,   1), (  90,  -90,-20)),

        ((   0,   1,   0), (   0,   0, -20)),
        ((  -1,   0,   0), (  90,   0, -20)),
        ((   0,  -1,   0), ( 180,   0, -20)),
        ((   1,   0,   0), ( 270,   0, -20)),
        ((   0,   1,   0), ( 360,   0, -20)),
        ((   0,   0,  -1), (  90,   90,-20)),
        ((   0,  -1,   0), ( 180,   90,-20)),
        ((   0,   0,   1), (  90,  -90,-20)),
    ]

    def testGenerateGPS(self):
        import geometry
        iso = (20, -14, 55)
        test_cases_failed = 0
        for idx, (direction, (angle_gantry, angle_couch, angle_coll)) in enumerate(self.test_cases):
            source, focus, iso2 = geometry.calculate_gps_coordinates(
                position=[20, 20],
                angle_gantry=angle_gantry*torad,
                angle_couch=angle_couch*torad,
                angle_coll=angle_coll*torad,
                iso=iso,
                start=(-15,-234,-23),
                size=(256, 256, 100),
                spacing=(0.5, 0.5, 0.5),
                fmapdims=(40, 40),
                beamletspacing=(0.5, 0.5),
                beamletsize=(0.5, 0.5),
                sad=100.0,
            )
            beam_direction = normalize_vect(np.subtract(iso2, focus))
            try:
                np.testing.assert_almost_equal(beam_direction, direction, decimal=4, err_msg="failed for angle ({}, {}, {})".format(angle_gantry, angle_couch, angle_coll))
                status=bcolors.OKGREEN+'PASS'+bcolors.ENDC
            except Exception as e:
                test_cases_failed += 1
                status=bcolors.FAIL+'FAIL'+bcolors.ENDC
            logger.debug('{:3d} [{!s}]: angle: ({:6.1f}, {:6.1f}, {:6.1f})' \
                  ' || dir: ({:5.2f}, {:5.2f}, {:5.2f})'
                  ' || exp: ({:5.2f}, {:5.2f}, {:5.2f})'.format(
                      idx, status, angle_gantry, angle_couch, angle_coll,
                      *beam_direction, *direction
                  ))
        assert test_cases_failed == 0

    def testCalculateSourcePlaneRotation(self):
        import generate_input
        test_cases_failed = 0
        for idx, (direction, (angle_gantry, angle_couch, angle_coll)) in enumerate(self.test_cases):
            xp, yp = generate_input.calculate_plane_rotation(
                angle_gantry=angle_gantry*torad,
                angle_couch=angle_couch*torad,
                angle_coll=angle_coll*torad,
            )
            beam_direction = normalize_vect(np.cross(xp, yp))
            try:
                np.testing.assert_almost_equal(beam_direction, direction, decimal=4, err_msg="failed for angle ({}, {}, {})".format(angle_gantry, angle_couch, angle_coll))
                status=bcolors.OKGREEN+'PASS'+bcolors.ENDC
            except Exception as e:
                test_cases_failed += 1
                status=bcolors.FAIL+'FAIL'+bcolors.ENDC
            logger.debug('{:3d} [{!s}]: angle: ({:6.1f}, {:6.1f}, {:6.1f})' \
                  ' || dir: ({:5.2f}, {:5.2f}, {:5.2f})'
                  ' || exp: ({:5.2f}, {:5.2f}, {:5.2f})'.format(
                      idx, status, angle_gantry, angle_couch, angle_coll,
                      *beam_direction, *direction
                  ))
        assert test_cases_failed == 0



if __name__ == '__main__':
    unittest.main()
