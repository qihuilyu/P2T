import unittest
from os.path import join as pjoin

import numpy as np

from setup_tests import test_data
from mcdose import get_trained_model
from mcdose.tf_functions import tf_normalize
from mcdose.visualize import create_volume_dose_figure, save_figure_array

class TestVisualize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = np.load(pjoin(test_data, 'datasamples_10.npz'))
        cls.inputs = data['inputs']
        cls.labels = data['labels']
        cls.preds  = data['predictions']

    def test_plot_multislice(self):
        # normalize
        n_labels, n_inputs, n_preds = tf_normalize(self.labels, (self.labels, self.inputs[...,0,None], self.preds))

        # create image
        idx = 8
        figimg = create_volume_dose_figure(
            np.stack([
                np.stack([
                    n_labels[idx, sliceidx,:,:,0],
                    n_inputs[idx, sliceidx,:,:,0],
                    n_preds [idx, sliceidx,:,:,0],
                    n_preds [idx, sliceidx,:,:,0]-n_labels[idx,sliceidx,:,:,0]
                ], axis=0) for sliceidx in range(n_labels.shape[1])
            ], axis=0),
            dpi=200,
            col_labels=['label', 'input', 'predict', 'predict - label'],
        )
        save_figure_array(figimg, pjoin(test_data, 'fig_multislice.png'))


    def test_predict(self):
        model = get_trained_model(config=pjoin(test_data, 'model_config.yml'),
                                  weights=pjoin(test_data, 'model_weights.hdf5'),
                                  )
        predictions = model.predict(self.inputs, batch_size=30)
        #  np.savez(pjoin(test_data, 'datasamples_10.npz'),
        #           inputs=self.inputs,
        #           labels=self.labels,
        #           predictions=predictions
        #  )


if __name__ == '__main__':
    unittest.main()
