import sys, os
from os.path import join as pjoin
from collections import OrderedDict
import logging
import math
import copy
import shutil

from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import numpy as np

from . import metrics, losses, callbacks
from .models.unetdenoiser import pad_for_unet, unpad_from_unet

logger = logging.getLogger("MCDose."+__name__)

voxelsize=0.25  # TODO: load from dataset metadata

def train(model, train_dataloader, val_dataloader, config, iopaths, distribute_strategy, gamma_freq=20, tflogdir='logs'):
    valfrac = len(val_dataloader) / (len(train_dataloader) + len(val_dataloader))
    logger.info('Partition of Training/Validation Data ({:0.0%}:{:0.0%})'.format(1-valfrac, valfrac))

    # create smaller validation dataset for expensive evaluation
    subval_dataset = [val_dataloader.get_batch(10)]
    val_dataloader.reset()

    filewriter = tf.summary.create_file_writer(pjoin(tflogdir, 'custom'))
    image_filewriter = tf.summary.create_file_writer(pjoin(tflogdir, 'images'))
    filewriter.set_as_default()

    # define callbacks
    fit_callbacks = [
        keras.callbacks.TensorBoard(pjoin(tflogdir, 'metrics'), update_freq='batch', profile_batch=0, histogram_freq=5, write_graph=True),
        keras.callbacks.ModelCheckpoint(pjoin(iopaths.checkpointdir, 'weights.hdf5'), save_freq='epoch', save_weights_only=True),
        callbacks.TensorBoard(filewriter),
        callbacks.ImageLoggerCallback(subval_dataset, image_filewriter, eval_freq=5),
    ]
    if gamma_freq >= 1:
        fit_callbacks.append(
            callbacks.GammaAnalysisCallback(subval_dataset, voxelsize, filewriter, image_filewriter, eval_freq=gamma_freq),
        )

    # train/evaluate
    logger.info("Beginning the Training Procedure...")
    model.fit(
        train_dataloader,
        epochs=config['nepochs'],
        steps_per_epoch=config.get('steps_per_epoch', None),
        verbose=1,
        shuffle=False, # handled by dataloader already
        validation_data=val_dataloader,
        validation_steps=None,
        callbacks=fit_callbacks,
        workers=1,
    )

def test(model, test_dataloader, config, iopaths, distribute_strategy, test_plots=False, gamma=False, baseline=False):
    logger.debug('Testing'+(' (baseline)' if baseline else '')+'...')
    resultsdir = iopaths.resultsdir + ('-baseline' if baseline else '')
    os.makedirs(resultsdir, exist_ok=True)

    # perform standard testing with metrics
    model.reset_metrics()
    results = model.evaluate(
        test_dataloader,
        verbose=1,
    )
    try:
        len(results)
    except:
        results = [results]
    results = {model.metrics_names[i]: results[i] for i in range(len(results))}

    if gamma:
        noop_filewriter = tf.summary.create_noop_writer()
        gamma_cb = callbacks.GammaAnalysisCallback(test_dataloader, voxelsize, noop_filewriter, noop_filewriter, baseline=baseline)
        gamma_cb.model = model
        gamma_results = gamma_cb.evaluate(0, mode='testing')
        results = {**results, **gamma_results}

    avg_results = OrderedDict()
    for k, v in results.items():
        avg_results[k] = avg_results.get(k, 0) + v

    testlogger = logging.getLogger('MCDose.'+__name__+'.test')
    testlogger.addHandler( logging.FileHandler(pjoin(resultsdir, 'avg_results.txt'), mode='w') )
    testlogger.addHandler( logging.StreamHandler() )
    testlogger.propagate = False
    testlogger.setLevel(logging.DEBUG)
    for k, v in avg_results.items():
        testlogger.info('Avg - {:s}: {}'.format(k, v))
    del testlogger.handlers[:]

    return avg_results
#=============================================================

def train_step_factory(model, optimizer, loss_fn):
    @tf.function
    def step_fn(inputs, labels, weights=None):
        with tf.GradientTape() as tape:
            # pad to make product of 2^n for some n (so downsampling always works)
            pinputs, orig_size = pad_for_unet(inputs, nscales=3)
            preds = model(pinputs, training=True)
            preds = unpad_from_unet(preds, orig_size)

            loss = loss_fn(labels, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    return step_fn

def val_step_factory(model, loss_fn):
    @tf.function
    def step_fn(inputs, labels, weights=None):
        # pad to make product of 2^n for some n (so downsampling always works)
        pinputs, orig_size = pad_for_unet(inputs, nscales=3)
        preds = model(pinputs, training=False)
        preds = unpad_from_unet(preds, orig_size)

        loss = loss_fn(labels, preds)
        return loss
    return step_fn

@tf.function
def distributed_step(step_fn, distribute_strategy, inputs, labels, weights=None):
    with distribute_strategy.scope():
        per_replica_losses = distribute_strategy.experimental_run_v2(step_fn, args=(inputs, labels, weights))
        return distribute_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

def train_custom(model, train_dataloader, val_dataloader, config, iopaths, distribute_strategy, gamma_freq, tflogdir):
    valfrac = len(val_dataloader) / (len(train_dataloader) + len(val_dataloader))
    logger.info('Partition of Training/Validation Data ({:0.0%}:{:0.0%})'.format(1-valfrac, valfrac))

    # create smaller validation dataset for expensive evaluation
    subval_dataset = [val_dataloader.get_batch(10)]
    val_dataloader.reset()

    # distribute datasets
    train_dist_dataset = distribute_strategy.experimental_distribute_dataset(
        tf.data.Dataset.from_generator(lambda: train_dataloader, output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape((None, None, None, None, None)), tf.TensorShape((None, None, None, None, None))))
    )
    val_dist_dataset = distribute_strategy.experimental_distribute_dataset(
        tf.data.Dataset.from_generator(lambda: val_dataloader, output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape((None, None, None, None, None)), tf.TensorShape((None, None, None, None, None))))
    )

    os.makedirs(pjoin(tflogdir, 'metrics'), exist_ok=True)
    train_filewriter = tf.summary.create_file_writer(pjoin(tflogdir, 'metrics', 'train'))
    val_filewriter = tf.summary.create_file_writer(pjoin(tflogdir, 'metrics', 'validation'))
    filewriter = tf.summary.create_file_writer(pjoin(tflogdir, 'custom'))
    image_filewriter = tf.summary.create_file_writer(pjoin(tflogdir, 'images'))
    filewriter.set_as_default()

    # prepare loss function
    #  loss_fn = losses.WeightedLoss(weight_cb=losses.threshold_weight(thresh=0.05, mask_cval=0.001), loss_cb=losses.meanabsoluteerror)
    #loss_fn = losses.WeightedLoss(weight_cb=losses.exponential_weight(decay_rate=2.5), loss_cb=losses.meanabsoluteerror)
    #  loss_fn = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM)
    loss_fn = losses.WeightedLoss(loss_cb=losses.meansquarederror, weight_cb=losses.exponential_weight(decay_rate=3.0))
    optimizer = model.optimizer
    train_step_fn = train_step_factory(model, optimizer, loss_fn)
    val_step_fn = val_step_factory(model, loss_fn)

    # checkpoint savers
    weights_save_path = pjoin(iopaths.checkpointdir, 'weights.hdf5')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkmanager = tf.train.CheckpointManager(checkpoint, directory=iopaths.checkpointdir,
                                              max_to_keep=5)

    fit_callbacks = [
        callbacks.ImageLoggerCallback(subval_dataset, image_filewriter, eval_freq=5),
        callbacks.TensorBoard(filewriter),
    ]
    for cb in fit_callbacks:
        cb.model = model
    fit_metrics = [
        keras.metrics.MeanAbsoluteError(),
        keras.metrics.MeanSquaredError(),
    ]

    # log model graph to tensorboard
    tf.summary.trace_on(graph=True, profiler=False)
    val_step_fn(*val_dataloader[0])
    val_dataloader.reset()
    with train_filewriter.as_default():
        tf.summary.trace_export('Default', step=0)

    step = 0
    train_steps_per_epoch = min(len(train_dataloader), config.get('steps_per_epoch', len(train_dataloader)))
    val_steps_per_epoch   = min(len(val_dataloader),   config.get('steps_per_epoch', len(val_dataloader)))
    for epoch in tqdm(range(1, config['nepochs']+1), desc='Training (epochs)'):
        # TRAINING LOOP - batched
        #==========================
        total_loss = 0
        with tqdm(train_dist_dataset, total=train_steps_per_epoch, desc='Training (batches)') as train_pbar:
            for nbatches, batched_examples in enumerate(train_pbar):
                if nbatches >= train_steps_per_epoch:
                    break
                if len(batched_examples) == 2:
                    inputs, labels = batched_examples
                    weights=None
                else:
                    inputs, labels, weights = batched_examples

                batch_loss = distributed_step(train_step_fn, distribute_strategy, inputs, labels, weights)
                total_loss += batch_loss
                train_pbar.set_postfix(batch_loss=batch_loss.numpy())

                with train_filewriter.as_default():
                    tf.summary.scalar('batch_loss', batch_loss, step)

                for cb in fit_callbacks:
                    cb.on_train_batch_end(nbatches)

                step += 1
        train_loss = total_loss / nbatches
        #==========================

        # VALIDATION LOOP - batched
        #==========================
        total_loss = 0
        with tqdm(val_dist_dataset, total=val_steps_per_epoch, desc='Validation (batches)') as val_pbar:
            for nbatches, batched_examples in enumerate(val_pbar):
                if nbatches >= train_steps_per_epoch:
                    break
                if len(batched_examples) == 2:
                    inputs, labels = batched_examples
                    weights=None
                else:
                    inputs, labels, weights = batched_examples

                batch_loss = distributed_step(val_step_fn, distribute_strategy, inputs, labels, weights)
                total_loss += batch_loss
                val_pbar.set_postfix(batch_loss=batch_loss.numpy())
        val_loss = total_loss / nbatches
        #==========================

        # REPORTING
        for cb in fit_callbacks:
            cb.on_epoch_end(epoch)

        with train_filewriter.as_default():
            tf.summary.scalar("epoch_loss", train_loss, epoch-1)
        with val_filewriter.as_default():
            tf.summary.scalar("epoch_loss", val_loss, epoch-1)
        logger.info('Epoch {}, Loss: {}, Val Loss: {}'.format(epoch, train_loss, val_loss))

        # Save model weights (checkpoint)
        if epoch%5 == 0:
            model.save_weights(weights_save_path, save_format='h5')
            checkmanager.save()
        if epoch%20 == 0:
            base, ext =  os.path.splitext(weights_save_path)
            copy_weights_path = base+"_epoch{:04d}".format(epoch)+ext
            shutil.copy2(weights_save_path, copy_weights_path)


def test_custom(model, test_dataloader, config, iopaths, distribute_strategy, test_plots=False, gamma=False, baseline=False):
    logger.debug('Testing'+(' (baseline)' if baseline else '')+'...')
    resultsdir = iopaths.resultsdir + ('-baseline' if baseline else '')
    os.makedirs(resultsdir, exist_ok=True)

    test_metrics = [
        keras.metrics.MeanAbsoluteError(),
        keras.metrics.MeanSquaredError(),
        metrics.NMAEMetric(),
        metrics.MaskedNMAEMetric(threshold=0.01)
    ]
    for metric in test_metrics:
        metric.reset_states()

    tf.config.experimental_run_functions_eagerly(True)
    # TESTING LOOP - batched
    #==========================
    total_loss = 0
    with tqdm(test_dataloader, total=len(test_dataloader), desc='Testing (batches)') as test_pbar:
        for nbatches, batched_examples in enumerate(test_pbar):
            if nbatches > 20:
                break
            if len(batched_examples) == 2:
                inputs, labels = batched_examples
                weights=None
            else:
                inputs, labels, weights = batched_examples

            # prediction
            pinputs, orig_size = pad_for_unet(inputs, nscales=3)
            preds = model(pinputs, training=False)
            preds = unpad_from_unet(preds, orig_size)

            for metric in test_metrics:
                metric.update_state(labels, preds)
            test_pbar.set_postfix({metric.name: metric.result().numpy() for metric in test_metrics})
    #==========================

    # perform standard testing with metrics
    results = {metric.name: metric.result().numpy() for metric in test_metrics}

    if gamma:
        noop_filewriter = tf.summary.create_noop_writer()
        gamma_cb = callbacks.GammaAnalysisCallback(test_dataloader, voxelsize, noop_filewriter, noop_filewriter, baseline=baseline)
        gamma_cb.model = model
        gamma_results = gamma_cb.evaluate(0, mode='testing')
        results = {**results, **gamma_results}

    avg_results = OrderedDict()
    for k, v in results.items():
        avg_results[k] = avg_results.get(k, 0) + v

    testlogger = logging.getLogger('MCDose.'+__name__+'.test')
    testlogger.addHandler( logging.FileHandler(pjoin(resultsdir, 'avg_results.txt'), mode='w') )
    testlogger.addHandler( logging.StreamHandler() )
    testlogger.propagate = False
    testlogger.setLevel(logging.DEBUG)
    for k, v in avg_results.items():
        testlogger.info('Avg - {:s}: {}'.format(k, v))
    del testlogger.handlers[:]

    test_dataloader.reset()

    return avg_results
