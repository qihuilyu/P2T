import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import gc
from collections import OrderedDict

from . import gamma
from .tf_functions import tf_normalize
from .visualize import plot_dose, plot_gamma_summary, plot_gamma, plot_gamma_components, plot_gamma_scatter, plot_profile, register_custom_scalars_layout, tile, MidpointNormalize, create_volume_dose_figure
from .models.unetdenoiser import pad_for_unet, unpad_from_unet

logger = logging.getLogger('MCDose.'+__name__)

class GammaAnalysisCallback(keras.callbacks.Callback):
    def __init__(self, dataloader, voxelsize, scalar_filewriter, image_filewriter, eval_freq=1, baseline=False, model=None):
        super().__init__()
        self.scalar_filewriter = scalar_filewriter
        self.image_filewriter = image_filewriter
        self.dataloader = dataloader
        self.voxelsize = voxelsize
        self.eval_freq = eval_freq
        self.baseline = baseline
        if model is not None:
            self.model = model

    def on_epoch_end(self, epoch, logs=None):
        """Perform after each epoch during training only"""
        if epoch % self.eval_freq == 0:
            with self.scalar_filewriter.as_default():
                self.evaluate(epoch*self.params['steps'], mode='validation')

    def evaluate(self, iter_num, mode='validation'):
        """Iterate over all batches in dataloader and perform gamma analysis on every example"""
        results = OrderedDict()

        # run prediction
        inputs = []
        labels = []
        predictions = []
        for batch_inputs, batch_labels in self.dataloader:
            # enforce 4d shape
            if len(batch_inputs) == 1:
                batch_inputs = batch_inputs[None, :, :, :]
                batch_labels = batch_inputs[None, :, :, :]
            inputs.append(batch_inputs)
            labels.append(batch_labels)
            predictions.append(
                self.model.predict(
                    batch_inputs,
                    batch_size = len(batch_inputs),
            ))
        inputs = np.concatenate(inputs, axis=0)
        labels = np.concatenate(labels, axis=0)
        predictions = np.concatenate(predictions, axis=0)

        # normalization
        n_labels, n_inputs, n_predictions = tf_normalize(labels, (labels, inputs[:,:,:,0,None], predictions))

        # perform gamma analyses
        gamma_params = [
            {'dta': 0.1, 'dd': 0.001},
            {'dta': 0.2, 'dd': 0.002},
            #  {'dta': 0.5, 'dd': 0.005},
            #  {'dta': 1.0, 'dd': 0.01},
            #  {'dta': 2.0, 'dd': 0.02}
        ]
        gamma_pass_sums = [0 for _ in range(len(gamma_params))]
        gamma_pass_masked_sums = [0 for _ in range(len(gamma_params))]
        for idx in range(len(n_inputs)):
            logger.info("Evaluating {} of {}".format(idx, len(n_inputs)))
            _label = n_labels[idx,:,:,0]
            _input = n_inputs[idx,:,:,0]
            _prediction = n_predictions[idx,:,:,0]

            # gamma analysis (note: dta is in mm, dd in fraction of max; as in 2mm/2%)
            if gamma:
                for gg, gparams in enumerate(gamma_params):
                    gamma_tag = '{:0.1f}mm_{:0.1%}/{!s}{:d}'.format(gparams['dta'], gparams['dd'], mode, idx+1)

                    gamma_map, dd_map, dta_map = gamma.gamma_analysis(_label, _prediction, gparams['dd'], gparams['dta'], self.voxelsize, output_components=True)
                    passing, total = gamma.gamma_passing_rate(gamma_map)
                    passing_rate = passing/total
                    gamma_pass_sums[gg] += passing_rate
                    tag = 'eval-gammapass/'+gamma_tag
                    with self.scalar_filewriter.as_default():
                        tf.summary.scalar(tag, passing_rate, step=iter_num)
                    with self.image_filewriter.as_default():
                        tf.summary.image(tag, plot_gamma(gamma_map, colorbar=True, annotate='{:0.2%}'.format(passing_rate)), step=iter_num)
                        #  tf.summary.image('gammacomponents/'+gamma_tag,   plot_gamma_components(dd_map, dta_map, colorbar=True), iter_num)
                        #  tf.summary.image('gammascatter/'+gamma_tag,      plot_gamma_scatter(dd_map, dta_map, gamma_map, gparams['dd'], gparams['dta'], colorbar=True), iter_num)
                        #  tf.summary.image('gammapass-summary/'+gamma_tag, plot_gamma_summary(dd_map, dta_map, gamma_map, gparams['dd'], gparams['dta'], colorbar=True, annotate='{:0.2%}'.format(passing_rate)), iter_num)
                    logger.debug("{!s}: {:0.4%}".format(tag, passing_rate))

                    # mask based on dose value
                    mask_thresh = 0.10 # 10 percent of low variance max dose
                    mask = np.logical_or(_label>=mask_thresh, _prediction >= mask_thresh)
                    masked_passing, masked_total = gamma.gamma_passing_rate(gamma_map, mask)
                    masked_passing_rate = masked_passing / masked_total
                    gamma_pass_masked_sums[gg] += masked_passing_rate
                    partial_gamma_map = np.copy(gamma_map)
                    partial_gamma_map[~mask] = -1
                    masktag = 'eval-gammapass-masked/'+gamma_tag
                    tf.summary.scalar(masktag, masked_passing_rate, iter_num)
                    with self.image_filewriter.as_default():
                        tf.summary.image(masktag, plot_gamma(partial_gamma_map, colorbar=True, annotate='{:0.2%}'.format(masked_passing_rate)), iter_num)
                    logger.debug("{!s}: {:0.4%} ".format(masktag, masked_passing_rate))

        if gamma:
            for gparams, passing_sum, masked_passing_sum in zip(gamma_params, gamma_pass_sums, gamma_pass_masked_sums):
                gamma_tag = '{:0.1f}mm_{:0.1%}/{!s}'.format(gparams['dta'], gparams['dd'], mode)

                with self.scalar_filewriter.as_default():
                    passing_avg = passing_sum / len(n_inputs)
                    tf.summary.scalar('avg-eval-gammapass/'+gamma_tag, passing_avg, iter_num)
                    logger.info("--- {!s} ---- Average gamma ({:0.1f}mm/{:0.1%}): {:0.4%} ---".format(mode, gparams['dta'], gparams['dd'], passing_avg))

                    masked_passing_avg = masked_passing_sum / len(n_inputs)
                    tf.summary.scalar('avg-eval_gammapass-masked/'+gamma_tag, masked_passing_avg, iter_num)
                    logger.info("--- {!s} ---- Average gamma (masked) ({:0.1f}mm/{:0.1%}): {:0.4%} ---".format(mode, gparams['dta'], gparams['dd'], masked_passing_avg))

        self.scalar_filewriter.flush()
        self.image_filewriter.flush()
        plt.close('all')
        gc.collect(2)
        return results


class ImageLoggerCallback(keras.callbacks.Callback):
    def __init__(self, dataloader, image_filewriter, eval_freq=5):
        super().__init__()
        self.dataloader = dataloader
        self.image_filewriter = image_filewriter
        self.eval_freq = eval_freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.eval_freq == 0:
            with self.image_filewriter.as_default():
                try:
                    step = epoch*self.params['steps']
                except:
                    step = epoch
                self.evaluate(step)

    def evaluate(self, iter_num):
        mode = 'validation'

        # run prediction
        inputs, labels = self.dataloader[0][:2]
        pinputs, orig_size = pad_for_unet(inputs, nscales=3)
        predictions = self.model.predict(
            tf.convert_to_tensor(pinputs),
            batch_size=len(inputs),
        )
        predictions = unpad_from_unet(predictions, orig_size)

        # normalization
        n_labels, n_inputs, n_predictions = tf_normalize(labels, (labels, inputs[:,:,:,:,0,None], predictions))

        # log images
        with self.image_filewriter.as_default():
            for idx in range(len(n_inputs)):
                tf.summary.image(
                    'multi_slice_composite/{!s}{:d}'.format(mode, idx+1),
                    create_volume_dose_figure(
                        np.stack([
                            np.stack([
                                n_labels[idx, sliceidx,:,:,0],
                                n_inputs[idx, sliceidx,:,:,0],
                                n_predictions[idx, sliceidx,:,:,0],
                                n_predictions[idx, sliceidx,:,:,0]-n_labels[idx,sliceidx,:,:,0]
                            ], axis=0) for sliceidx in range(n_labels.shape[1])
                        ], axis=0),
                        dpi=200,
                        col_labels=['label', 'input', 'predict', 'predict - label'],
                    ),
                    iter_num
                )


class TensorBoard(keras.callbacks.Callback):
    def __init__(self, filewriter):
        super().__init__()
        self.filewriter = filewriter
        self.step = 0

    def on_train_begin(self, logs=None):
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1

    def on_epoch_end(self, epoch, logs=None):
        with self.filewriter.as_default():
            step = self.step
            lr = self.model.optimizer.lr
            if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
                lr = lr(step)
            tf.summary.scalar('learning_rate', lr, step=step)
