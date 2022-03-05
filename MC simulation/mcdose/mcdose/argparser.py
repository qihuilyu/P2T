import argparse

from .utils import limited_float

parser = None

def parse_args(*args, **kwargs):
    if parser is None:
        _init_parser()
    return parser.parse_args(*args, **kwargs)

def _init_parser():
    global parser
    parser = argparse.ArgumentParser(description='Deep monte carlo dose denoiser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Execution Actions
    subparsers = parser.add_subparsers(title='Execution Actions', help='select execution action', dest='action')
    parser.add_argument('-L', '--loglevel', default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL'], help='set the logging level' )

    # Training/Validation Settings
    train_parser = subparsers.add_parser('train', help='Initiate model training')
    add_org_arg_group(train_parser)
    add_config_override_group(train_parser)
    train_parser.add_argument('--resume', action='store_true', help='attempt to load model checkpoint and continue training')
    train_parser.add_argument('--noval', action='store_true', help='skip all evaluation steps')
    train_parser.add_argument('--gamma', action='store_true', help='run all gamma evaluation steps')
    train_parser.add_argument('--debug', action='count')

    # Testing Settings
    test_parser = subparsers.add_parser('test', help='Initiate model testing')
    add_org_arg_group(test_parser, run_required=True)
    add_config_override_group(test_parser)
    test_parser.add_argument('--test-plots', action='store_true', help='produce plots during test')
    test_parser.add_argument('--gamma', action='store_true', help='run all gamma evaluation steps')

    # Prediction Settings
    predict_parser = subparsers.add_parser('predict', help='Initiate model prediction')
    add_config_override_group(predict_parser)
    predict_parser.add_argument('--in', '-i', dest='data_in', type=str, nargs="+", help='input filename(s) to use in prediction/testing')
    predict_parser.add_argument('--size', '-s', dest='data_size', type=int, nargs='+', help='size of input as x y z')
    predict_parser.add_argument('--out', '-o', dest='pred_out', type=str, default='dosepred.bin', help='output (predicted dose) filename')

    # Computational Settings
    comp_parser = parser.add_argument_group('Computation Settings', 'change the way actions are executed')
    comp_parser.add_argument('--gpu', type=str, default='0', help='select GPU # to use (# will not be validated)')
    comp_parser.add_argument('--cpu', action='store_true', help='disable GPU processing')
    comp_parser.add_argument('--cache-size', type=int, default=3, help='number of data example files to keep in cache before evicting')
    comp_parser.add_argument('--cache-only', action='store_true', help='limit dataset to only what can fit in memory cache (set with --cache-size)')
    comp_parser.add_argument('--seed', type=int, help='set the random number seed')

    # Model Definition Settings
    model_parser = parser.add_argument_group('Model Settings', 'change the model definition')
    model_parser.add_argument('--config', type=str, default='config.yml', help='configuration file defining model/execution settings')

def add_org_arg_group(parser, run_required=False):
    org_parser = parser.add_argument_group('Organization Settings', 'setup and select data organization')
    org_parser.add_argument('--datadir', help='data *.npy files are stored here in dirs named for each doi')
    org_parser.add_argument('--rundir', default='runs', help='path to dir containing all runs')
    org_parser.add_argument('--run', type=int, required=run_required, default=None, help='name of this run (if exists, reload checkpoint and append to log; else create new run dir)')
    org_parser.add_argument('--memo', type=str, default=None, help='add memo identify results and logs in existing run folder')
    org_parser.add_argument('--clean-runs', action='store_true', help='Clean runs from <rundir> that don\'t contain model checkpoints')

def add_config_override_group(parser):
    over_parser = parser.add_argument_group('Config Override Settings', 'override some settings found in the config.yml file for convenience')
    over_parser.add_argument('--batch-size', type=int, default=None, help='Override number images in batch set in config')
    over_parser.add_argument('--nepochs', type=int, default=None, help='# of epoch')
    over_parser.add_argument('--lr', dest='learning_rate', type=float, default=None, help='initial learning rate for adam')
    over_parser.add_argument('--lr-decay', dest='learning_rate_decay', type=float, default=None, help='exponential decay rate')
    over_parser.add_argument('--steps', '--steps-per-epoch', dest='steps_per_epoch', type=int, default=None, help='limit training epochs to fewer batches')
