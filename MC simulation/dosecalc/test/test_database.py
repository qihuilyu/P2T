import unittest
import sys
import os
from os.path import join as pjoin
import math

from bson import ObjectId

os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(pjoin(os.pardir, 'webapi')))

from api_enums import VARTYPE, STORAGETYPE
import database

class TestMCSource(unittest.TestCase):

    @staticmethod
    def _gen_simdoc():
        doc = database.gen_doc_simulation(
            image_id=ObjectId(),
            geom_id=ObjectId(),
            beam_id=ObjectId(),
            subbeam_id=ObjectId(),
            vartype=VARTYPE.LOW,
            num_runs=1,
            num_particles=100,
            magnetic_field=(0,0,0,'tesla'),
            storage_type=STORAGETYPE.SPARSE,
            callargs=['--sparse', 'extra args 1', 'extra args 2'],
        )
        return doc

    def test_generate_simulation_payload(self):
        doc = self._gen_simdoc()
        payload = database.generate_simulation_payload()
        print(doc)

if __name__ == '__main__':
    unittest.main()
