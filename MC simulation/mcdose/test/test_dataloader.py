import unittest
from os.path import join as pjoin

from setup_tests import test_data
from mcdose import dataloader

class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = pjoin(test_data, 'small.npy')
        cls.dl = dataloader.DataLoaderNpyFiles([cls.f], batch_size=1)


    def test_get_npy_files(self):
        shape = dataloader.DataLoaderNpyFiles.get_npy_array_shape(pjoin(test_data, 'small.npy'))
        self.assertEqual(shape, (3,5,5,4))

    def test___init__(self):
        self.assertEqual(self.dl.num_examples_in_file[self.f], 3)



if __name__ == '__main__':
    unittest.main()
