import sys
import os
import json
import shutil
from os.path import join as pjoin
import logging
from collections import namedtuple
from datetime import datetime
from pprint import pprint, pformat

import absl.logging
import numpy as np
import scipy.stats
import tensorflow as tf
from tensorflow import keras

from . import actions, argparser, weights, losses
from .dataloader import DataLoaderNpyFiles
from .models.unetdenoiser import UNetDenoiser, UNetDenoiserStatic, ModelWrapper, pad_for_unet, unpad_from_unet
from .utils import get_unique_run_name, load_bin, save_bin, save_config, load_config, combine_config
from .tf_functions import tf_normalize, tf_log

#  tf.autograph.set_verbosity(6, alsologtostdout=True)
#  tf.debugging.set_log_device_placement(True)

timestamp = datetime.now().strftime("%Y-%m-%d_%T")

logger = logging.getLogger("MCDose."+__name__)
IOPaths = namedtuple('IOPaths', ('rundir', 'logdir', 'checkpointdir',
                                 'tflogdir', 'resultsdir', 'datadir'))


def main():
    # configuration common to all "actions"
    ## workaround to intrusive abseil logger in TF2.0 (double logging)
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    rootlogger = logging.getLogger('MCDose')
    rootlogger.setLevel(args.loglevel)
    rootlogger.addHandler( logging.StreamHandler() )

    if args.seed:
        np.random.seed(args.seed)

    distribute_strategy = None
    if args.cpu:
        # set GPU restrictions
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        logger.debug("Processing Mode: CPU\n")
        distribute_strategy = default_distribute_strategy(cpu=True)
    else:
        # set GPU restrictions
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        logger.debug("Processing Mode: GPU (#{})\n".format(args.gpu))
        distribute_strategy = default_distribute_strategy()

    # check eager mode
    logger.info('Tensorflow is running in {} mode'.format(
        'EAGER' if tf.executing_eagerly() else "GRAPH"
    ))

    # initialize global runs folder
    allrunsdir = args.rundir
    os.makedirs(allrunsdir, exist_ok=True)

    # One-Off Actions
    if args.clean_runs:
        clean_runs(allrunsdir)
        sys.exit(0)

    # decide on this run's name
    rundir = None
    if args.run is not None:
        # trying to specify an existing run...match on integer, not string
        for d in os.listdir(allrunsdir):
            try:
                if os.path.isdir(pjoin(allrunsdir, d)) and int(d)==int(args.run):
                    rundir = d
                    break
            except: pass

    if rundir is None:
        rundir = get_unique_run_name(allrunsdir)
    rundir = pjoin(allrunsdir, rundir)

    # configure run paths with optional memo
    memo = (('-'+args.memo) if args.memo else '')
    rundir = rundir+memo
    iopaths = IOPaths(**{
        'rundir':        rundir,
        'logdir':        pjoin(rundir, 'logs'),
        'checkpointdir': pjoin(rundir, 'checkpoints'),
        'tflogdir':      pjoin(rundir, 'tflogs'),
        'resultsdir':    pjoin(rundir, 'results' + memo),
        'datadir':       args.datadir,
    })
    # ensure output directories exist
    os.makedirs(rundir, exist_ok=True)
    for d in (iopaths.checkpointdir,
              iopaths.logdir,
              iopaths.tflogdir,
              iopaths.resultsdir):
        os.makedirs(d, exist_ok=True)

    # combine cmdline args, general config, and run config (if existing run is used)
    try:
        config = load_config(iopaths.rundir)
    except FileNotFoundError:
        config = load_config(args.config)
    config = combine_config(
        {
            'timestamp': timestamp,
            'datadir': iopaths.datadir,
            'seed': args.seed,
            'memo': args.memo,
        },
        config,
    )
    # override config settings with cmdline args
    config = combine_config(
        {
            'batch_size': vars(args).get('batch_size', None),
        },
        config,
    )
    save_config(iopaths.rundir, config)

    # initiate logging
    setup_logging(args.action, iopaths)

    # launch selected action
    actionfunc = globals()[args.action]
    if callable(actionfunc):
        actionfunc(config, iopaths, distribute_strategy)
    else:
        raise ValueError('Action "{}" is not supported'.format(args.action))
        sys.exit(1)

def setup_logging(action, iopaths):
    # deterimine runtime state
    action = args.action

    # finish logging setup
    logfilename = pjoin(iopaths.logdir, 'log-'+action+'.txt')
    filehandler = logging.FileHandler(logfilename)
    rootlogger = logging.getLogger('MCDose')
    rootlogger.addHandler( filehandler )
    tflogger = tf.get_logger()
    tflogger.addHandler(filehandler)

def train(config, iopaths, distribute_strategy):
    # override config settings with cmdline args
    config = combine_config(
        {
            'learning_rate':   vars(args).get('learning_rate',   None),
            'learning_rate_decay': vars(args).get('learning_rate_decay', None),
            'nepochs':         vars(args).get('nepochs',         None),
            'steps_per_epoch': vars(args).get('steps_per_epoch', None),
        },
        config,
    )

    # set defaults if they havent been set on cmdline or in config file
    config['batch_size'] = config.get('batch_size', 30)
    config['learning_rate'] = config.get('learning_rate', 1e-2)
    config['learning_rate_decay'] = config.get('learning_rate_decay', 0.985)

    save_config(iopaths.rundir, config)

    # setup loss function per-sample weighting
    # TODO: refactor into common selector class
    wfuncconfig = config['sample_weights']
    supported_wfuncs = {
        None: lambda **kwargs: None,
        'lin_norm_sum': weights.sample_lin_norm_sum,
        'exp_norm_sum': weights.sample_exp_norm_sum,
    }
    wfunctype = wfuncconfig['type']
    if wfunctype not in supported_wfuncs:
        raise ValueError('Loss weighting function "{}" is not supported. Must be one of {!s}'.format(wfunctype, list(supported_wfuncs.keys())))
    wfunc = supported_wfuncs[wfunctype](**wfuncconfig.get(wfunctype, {}))

    logger.info('Loading TRAINING data from "{}":'.format(iopaths.datadir))
    if not iopaths.datadir or not os.path.isdir(iopaths.datadir):
        raise RuntimeError('Training data folder (datadir) could not be located at "{}"'.format(iopaths.datadir))
    train_dataloader = DataLoaderNpyFiles.fromFolder(
        pjoin(iopaths.datadir, 'train'),
        config['batch_size'],
        weight_func=wfunc,
        full_batches_only=True,
        cache_size=args.cache_size,
        limit=args.cache_size if args.cache_only else None,
        randomization=DataLoaderNpyFiles.Randomization.Shuffle
    )
    val_dataloader = DataLoaderNpyFiles.fromFolder(
        pjoin(iopaths.datadir, 'validate'),
        config['batch_size'],
        weight_func=wfunc,
        full_batches_only=False,
        cache_size=args.cache_size,
        limit=args.cache_size if args.cache_only else None,
        randomization=DataLoaderNpyFiles.Randomization.NoShuffle
    )

    # copy normalization statistics if available in dataset
    statsfile = pjoin(iopaths.datadir, 'stats.json')
    if os.path.isfile(statsfile):
        shutil.copy2(statsfile, pjoin(iopaths.rundir, 'normstats.json'))

    if args.debug:
        ndebugbatches = 3
        logger.info('Only using {} batches for debugging'.format(ndebugbatches))
        train_dataloader.num_batches = ndebugbatches
        val_dataloader.num_batches = ndebugbatches

    # optionally, train on subset of dataset
    config['steps_per_epoch'] = max(1, min(len(train_dataloader),
        config.get('steps_per_epoch') if config.get('steps_per_epoch', None) is not None else len(train_dataloader)
        ))

    # is user hoping to load existing model, or requiring it?
    if args.resume:
        load_model = 'force'
    elif args.run:
        load_model = 'try'
    else:
        load_model = False

    model = create_model(config['model'], distribute_strategy, load_model=load_model, checkpointdir=iopaths.checkpointdir)
    prepare_model(model, train_dataloader, config, distribute_strategy=distribute_strategy)


    actions.train_custom(model, train_dataloader, val_dataloader, config, iopaths,
                         distribute_strategy, 0, iopaths.tflogdir)
    #  actions.train(**{
    #      'model':               model,
    #      'train_dataloader':    train_dataloader,
    #      'val_dataloader':      val_dataloader,
    #      'config':              config,
    #      'iopaths':             iopaths,
    #      'distribute_strategy': distribute_strategy,
    #      'gamma_freq':          (-1 if not args.gamma else 20),
    #      'tflogdir':            iopaths.tflogdir,
    #  })


def test(config, iopaths, distribute_strategy):
    """average over all test set slices"""
    # test trained model
    logger.info('Loading TESTING data:')
    test_dataloader = DataLoaderNpyFiles.fromFolder(pjoin(iopaths.datadir, 'test'), config['batch_size'], full_batches_only=False)

    # force testing on single GPU
    distribute_strategy = default_distribute_strategy(single_device=True)

    model = get_trained_model(
        config['model'],
        weights=iopaths.checkpointdir,
        normstats=pjoin(iopaths.rundir, 'normstats.json'),
        distribute_strategy=distribute_strategy,
    )

    inputs = keras.Input(shape=(None, None, None, 2))
    noop_model = keras.Model(inputs, inputs[...,0,None])

    for this_model, baseline in ((model, False), (noop_model, True)):
        #  actions.test(this_model, test_dataloader, config, iopaths, distribute_strategy, test_plots=args.test_plots,
        actions.test_custom(this_model, test_dataloader, config, iopaths, distribute_strategy, test_plots=args.test_plots,
                     baseline=baseline, gamma=args.gamma)


def predict(config, iopaths, distribute_strategy):
    assert len(args.data_in) > 0
    assert len(args.data_size) > 0

    model = get_trained_model(
        config['model'],
        weights=iopaths.checkpointdir,
        normstats=pjoin(iopaths.rundir, 'normstats.json'),
        distribute_strategy=distribute_strategy,
    )

    for ii, fname in enumerate(args.data_in):
        assert os.path.isfile(fname)
        inputs = load_bin(fname, args.data_size, add_channel_axis=True, norm=True)

        pred_dose = model.predict(inputs, batch_size=config['batch_size'])

        if len(args.data_in)>1:
            outname = args.pred_out+str(ii)+'.bin'
        else:
            outname = args.pred_out
        save_bin(outname, pred_dose[:,:,:,0] )


def create_model(modelconfig, distribute_strategy=None, load_model=False, checkpointdir=None):
    """ modelconfig is expected to be the subset of the full config, specific to the model """
    if distribute_strategy is None:
        distribute_strategy = default_distribute_strategy()

    # only proceed to create model if failed to load
    with distribute_strategy.scope():
        supported_models = {
            'static': ModelWrapper(UNetDenoiserStatic),
            'unet':  UNetDenoiser,
        }
        modeltype = modelconfig['type']
        if modeltype not in supported_models:
            raise ValueError('Model "{}" is not supported. Must be one of {!s}'.format(modeltype, list(supported_models.keys())))
        model = supported_models[modeltype](**modelconfig.get(modeltype, {}))

        if load_model:
            if os.path.isfile(checkpointdir):
                model_path = checkpointdir
            else:
                model_path = pjoin(checkpointdir, 'weights.hdf5')

            try:
                model.load_weights(model_path)
                logger.info('Loaded model from "{}"'.format(model_path))
            except Exception as e:
                if load_model == 'force':
                    logger.error('Failed to reload model checkpoint from "{}"'.format(model_path))
                    raise
                elif load_model == 'try':
                    logger.warning('Failed to reload model checkpoint from "{}". Randomly initializing model'.format(model_path))
                else:
                    logger.error('model load strategy "{!s}" is invalid. Must be one of {!s}'.format(load_model, ['force', 'try']))
                    sys.exit(1)

    return model

def prepare_model(model, dataloader, config, distribute_strategy):
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=float(config['learning_rate']),
        decay_steps=int(config.get('steps_per_epoch')) if config.get('steps_per_epoch', None) is not None else len(dataloader),
        decay_rate=float(config['learning_rate_decay']),
        staircase=False,
    )

    # configure optimizer
    optconfig = config['optimizer']
    supported_optimizers = {
        'sgd': keras.optimizers.SGD,
        'adam': keras.optimizers.Adam,
        'rmsprop': keras.optimizers.RMSprop,
    }
    opttype = optconfig['type']
    if opttype not in supported_optimizers:
        raise ValueError('Optimizer "{}" is not supported. Must be one of {!s}'.format(opttype, list(supported_optimizers.keys())))
    with distribute_strategy.scope():
        optimizer = supported_optimizers[opttype](**optconfig.get(opttype, {}), learning_rate=lr_schedule)

    # configure loss function
    lossconfig = config['loss']
    supported_losses = {
        'mse': keras.losses.MeanSquaredError,
        'mae': keras.losses.MeanAbsoluteError,
        'mse_tv': losses.MeanSquaredErrorTV,
    }
    losstype = lossconfig['type']
    if losstype not in supported_losses:
        raise ValueError('Loss function "{}" is not supported. Must be one of {!s}'.format(losstype, list(supported_losses.keys())))
    loss_function = supported_losses[losstype](**lossconfig.get(losstype, {}))

    # finalize model based on input data shape
    with distribute_strategy.scope():
        model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=[
                keras.metrics.MeanSquaredError(),
                keras.metrics.MeanAbsoluteError(),
                #  metrics.NMAEMetric(),
                #  metrics.MaskedNMAEMetric(threshold=0.2),
            ],
        )

        input_shape = dataloader[0][0].shape
        model.build((None, *input_shape[1:]))

    model.summary()
    return model

trained_model = None
def get_trained_model(config, weights, normstats=None, distribute_strategy=None):
    global trained_model
    if trained_model:
        return trained_model
    if isinstance(config, str):
        config = load_config(config)['model']
    if not isinstance(distribute_strategy, tf.distribute.Strategy):
        distribute_strategy = default_distribute_strategy()

    model = create_model(config, distribute_strategy, load_model='force', checkpointdir=weights)

    # Wrap model with normalization prestage matching that from training
    if normstats:
        if isinstance(normstats, str):
            with open(normstats, 'r') as fd:
                stats = json.load(fd)

        mean = tf.constant([stats['mean'][0], stats['mean'][2]], dtype=tf.float32, shape=(1,1,1,1,2))
        std  = tf.constant([stats['std'][0], stats['std'][2]],  dtype=tf.float32, shape=(1,1,1,1,2))
        inputs = keras.Input(shape=(None,None,None,2), dtype=tf.float32)
        x = (inputs-mean)/std # normalize

        x, orig_size = pad_for_unet(x, nscales=3)
        x = model(x)          # predict
        x = unpad_from_unet(x, orig_size)

        outputs = x*std[...,0,None]+mean[...,0,None]  # un-normalize
        model = keras.Model(inputs, outputs)
    trained_model = model
    return model

def default_distribute_strategy(cpu=False, single_device=False):
    if cpu:
        return tf.distribute.OneDeviceStrategy(device='/cpu:0')

    devices = tf.config.list_logical_devices("GPU")
    if not devices:
        return tf.distribute.OneDeviceStrategy(device='/cpu:0')
    elif devices and len(devices)>1 and not single_device:
        return tf.distribute.MirroredStrategy(devices=devices)
    elif devices and len(devices)==1 or single_device:
        return tf.distribute.OneDeviceStrategy(device=devices[0])

def clean_runs(allrunsdir):
    """Analyze each 'run' folder in 'allrunsdir' and delete those that have no valuable training data, such as
    a valid training checkpoint or tflogs files"""
    runs_to_delete = []
    for d in sorted(os.listdir(allrunsdir)):
        rundir = pjoin(allrunsdir, d)
        checkpoint_index = pjoin(rundir, 'checkpoints', 'checkpoint')
        weights_file = pjoin(rundir, 'checkpoints', 'weights.hdf5')
        if os.path.isfile(checkpoint_index) or os.path.isfile(weights_file):
            continue
        else:
            runs_to_delete.append(rundir)

    if not runs_to_delete:
        logger.info('Nothing to delete. exiting...')
        sys.exit(0)

    logger.warning("Runs to be deleted: \n{}".format(pformat(runs_to_delete)))
    user_response = input("Are you sure you want to delete these runs [y/N]: ")
    if not user_response.lower() in ('y', 'yes'):
        logger.warning('Aborted!')
    else:
        for rundir in runs_to_delete:
            try:
                shutil.rmtree(rundir)
                logger.info('Deleted run directory "{}"'.format(rundir))
            except Exception as e:
                logger.warning('Failed to delete run directory "{}"'.format(rundir))
                logger.debug('Error details: {!s}'.format(e))

#=================================================================================

# Entry point for CLI
if __name__ == "__main__":
    args = argparser.parse_args()
    main()
