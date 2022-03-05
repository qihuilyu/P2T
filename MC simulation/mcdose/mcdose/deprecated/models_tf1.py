import sys, os
from os.path import join as pjoin
import time
import datetime
import math
import re
import logging
import gc
from collections import OrderedDict

logger = logging.getLogger('MCDose.'+__name__)

from multi_gpu import assign_to_device, average_gradients
from utils import save_bin, save_as_image, tf_split_var, BatchConstructor, RandomBatchConstructor, augment_data, get_available_gpus
from visualize import plot_dose, plot_gamma_summary, plot_gamma, plot_gamma_components, plot_gamma_scatter, plot_profile, register_custom_scalars_layout, tile, MidpointNormalize
from tblogger import TBLogger
from resnet import *
from unet import make_unet, conv2d_block
import gamma
import metrics
import callbacks
from gamma import gamma_passing_rate, gamma_analysis

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorboard import summary as summary_lib

voxelsize = 2.5 # [unit: mm]

default_device='/gpu:0'


    def test(self, inputs, labels, results_dir, batch_size=32, baseline=False, plots=False, gamma=False):
        os.makedirs(results_dir, exist_ok=True)
        logger = logging.getLogger(__name__+'.'+self.__class__.__name__+'.test')
        logger.addHandler( logging.FileHandler(os.path.join(results_dir, 'results.txt'), mode='w') )
        logger.addHandler( logging.StreamHandler(sys.stdout) )
        logger.setLevel(logging.DEBUG)

        labels = labels.astype(np.float32)
        inputs = inputs.astype(np.float32)

        results = OrderedDict()

        predictions = np.empty_like(labels)
        metric_masked_mae = 0
        batchcon = BatchConstructor(inputs, labels)
        mark = 0
        nbatches = 0
        for (batch_inputs, batch_labels) in batchcon.iter_batches(batch_size):
            if baseline:
                predictions_ = batch_inputs[:,:,:,0,None]
                metric_masked_mae_ = self.sess.run(
                    self.metric_masked_mae,
                    feed_dict={self.labels: batch_labels,
                               self.inputs: batch_inputs,
                               self.predictions: batch_inputs[:,:,:,0, None],
                               self.is_training: False})
            else:
                predictions_, metric_masked_mae_ = self.sess.run(
                    [self.predictions, self.metric_masked_mae],
                    feed_dict={self.labels: batch_labels,
                               self.inputs: batch_inputs,
                               self.is_training: False})
            metric_masked_mae += metric_masked_mae_
            predictions[mark:mark+len(batch_inputs), :, :, :] = predictions_
            mark += len(batch_inputs)
            nbatches += 1
        metric_masked_mae /= nbatches

        results['Average Masked MAE'] = metric_masked_mae

        # normalization
        normgraph = tf.Graph()
        with normgraph.as_default():
            norm = tf_normalize(labels, (labels, inputs[:,:,:,0,None], predictions))
        normsess = tf.compat.v1.Session(graph=normgraph)
        n_labels, n_inputs, n_predictions = normsess.run( norm )
        normsess.close()

        if gamma:
            nplotslices = 30
            np.random.seed(1234)
            randidx = np.random.choice(range(len(labels)), (nplotslices,), replace=False)
            gamma_params = [{'dta': 0.1, 'dd': 0.001}, {'dta': 0.2, 'dd': 0.002}, {'dta': 0.5, 'dd': 0.005}, {'dta': 1.0, 'dd': 0.01}, {'dta': 2.0, 'dd': 0.02}]
            for gg, gparams in enumerate(gamma_params):
                dd = gparams['dd']
                dta = gparams['dta']

                passing = 0
                total = 0
                masked_passing = 0
                masked_total = 0
                gamma_map_slices = np.zeros((nplotslices, *labels.shape[1:3]))
                for ii, idx in enumerate(randidx):
                    _labels = labels[idx,:,:,0]
                    _inputs = inputs[idx,:,:,0]
                    _predictions = predictions[idx,:,:,0]

                    gamma_map = gamma_analysis(_labels, _predictions, dd, dta, voxelsize)
                    _passing, _total = gamma_passing_rate(gamma_map)
                    logger.info('slice{:05d}/{:05d} ({:05d}) | dta {:0.1f}mm | dd {:0.1%} | passing {:0.4%}'.format(ii+1, nplotslices, idx, dta, dd, _passing/_total))
                    passing += _passing
                    total += _total
                    gamma_map_slices[ii, :, :] = gamma_map

                    # mask based on dose value
                    mask_thresh = 0.10  # 10 percent of low variance max dose
                    mask = np.logical_or(_labels>=mask_thresh, _predictions>=mask_thresh)
                    _masked_passing, _masked_total = gamma_passing_rate(gamma_map, mask)
                    logger.info('slice{:05d}/{:05d} ({:05d}) | dta {:0.1f}mm | dd {:0.1%} | masked ({:0.1%} thresh) passing {:0.4%}'.format(ii+1, nplotslices, idx, dta, dd, mask_thresh, _masked_passing / _masked_total))
                    masked_passing += _masked_passing
                    masked_total += _masked_total

                passing_avg = passing/total
                res_key = 'Gamma Passing % (slices) ({:0.1f}mm_{:0.1%})'.format(dta, dd)
                results[res_key] = passing_avg

                masked_passing_avg = masked_passing/masked_total
                masked_res_key = 'Masked ({:0.0%} thresh) Gamma Passing % (slices) ({:0.1f}mm_{:0.1%})'.format(mask_thresh, dta, dd)
                results[masked_res_key] = masked_passing_avg

                if plots:
                    save_bin(os.path.join(results_dir, 'gamma2d_{:0.1f}mm_{:0.1%}.bin'.format(dta, dd)), gamma_map_slices)
                    tiled_gamma = tile([np.squeeze(x) for x in np.split(gamma_map_slices, len(gamma_map_slices))], 1, square=True, pad_width=0, pad_intensity=-1)
                    tiled_gamma = plot_gamma(tiled_gamma, colorbar=True, dpi=300)
                    save_as_image(os.path.join(results_dir, 'gamma2d_{:0.1f}mm_{:0.1%}.png'.format(dta, dd)), tiled_gamma, scale=3)

            if plots:
                for fname, arr in [ ('dose_lowvar', labels), ('dose_highvar', inputs), ('dose_pred', predictions), ('geometry', inputs[:,:,:,1, np.newaxis]), ('dose_difference', labels-predictions)]:
                    arr = arr[randidx,:,:,0]
                    save_bin(os.path.join(results_dir, fname+'.bin'), arr)
                    with open(os.path.join(results_dir, 'dims.txt'), 'w') as fd:
                        fd.write(' '.join(str(x) for x in list(arr.shape)))

                    vmin = 0
                    vmax = 1
                    cmap = 'viridis'
                    norm = None
                    if 'difference' in fname:
                        vmin = np.min(arr)
                        vmax = np.max(arr)
                        cmap = 'RdBu'
                        norm = MidpointNormalize(vmin=np.min(arr), vmax=np.max(arr), midpoint=0)
                    elif 'geometry' in fname:
                        vmin = 0.6
                        vmax = 1.35
                        cmap = 'gray'

                    tiled_arr = tile([np.squeeze(x) for x in np.split(arr, len(arr))], 1, square=True, pad_width=0, pad_intensity=-1)
                    tiled_arr = plot_dose(tiled_arr, colorbar=True, dpi=300, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
                    save_as_image(os.path.join(results_dir, fname+'.png'), tiled_arr, scale=3)

                #  for idx in range(len(labels)):
                #      _labels = labels[idx,:,:,0]
                #      _inputs = inputs[idx,:,:,0]
                #      _predictions = predictions[idx,:,:,0]

                #      save_bin_slices(os.path.join(results_dir, 'composite_dose_%d' % (idx + 1)),
                #               _labels.reshape(1,*_labels.shape),
                #               _inputs.reshape(1,*_inputs.shape),
                #               _predictions.reshape(1,*_predictions.shape))

        for k,v in results.items():
            logger.info('{}: {}'.format(k, v))

        logger.info('')
        logger.handlers = [] # reset logger

        return results
