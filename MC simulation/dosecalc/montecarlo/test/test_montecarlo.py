import unittest
import os
from os.path import join as pjoin
import shutil
import subprocess

os.chdir(os.path.abspath(os.path.dirname(__file__)))
from setup_tests import test_data


class TestMonteCarlo(unittest.TestCase):
    DEVNULL = None
    def setUp(self):
        self.DEVNULL = open(os.devnull, 'wb')
        if os.path.exists('build'):
            return
        print('Compiling Geant4...')
        subprocess.check_call(['bash', '-c', 'mkdir -p build; cd build; cmake ../..; make'], stdout=self.DEVNULL, stderr=self.DEVNULL)

    def tearDown(self):
        self.DEVNULL.close()
        #  try: shutil.rmtree('build')
        #  except: pass

    def testRunMonteCarlo(self):
        self.assertEqual(0, subprocess.call(['./build/dosecalc', '--outputdir', 'test_mc_output', '--individual', pjoin(test_data, 'mcgeo.txt'), pjoin(test_data, 'init.in'), pjoin(test_data, 'gps.mac'), pjoin(test_data, 'beamon.in')], stdout=self.DEVNULL, stderr=self.DEVNULL))
        for f in ['InputDensity.bin', pjoin('run000', 'dose3d.bin'), pjoin('run001', 'dose3d.bin')]:
            self.assertTrue(os.path.exists(pjoin('test_mc_output', f)))
        shutil.rmtree('test_mc_output')


if __name__ == '__main__':
    unittest.main()
