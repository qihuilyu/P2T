import main_ct as denoise
import math
from subprocess import run, CalledProcessError

import numpy as np
import numpy.random as nprand

import tensorflow as tf
from tensorflow.python.framework.errors_impl import ResourceExhaustedError, InvalidArgumentError

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=1, help='first trial index for logging')
parser.add_argument('--trials', type=int, default=50, help='number of trials to run')


def randrange(start, stop, size=None):
    return (stop-start)*nprand.random_sample(size) + start

if __name__ == "__main__":
    args = parser.parse_args()

    for trial in range(args.start, args.start+args.trials+1):
        tf.compat.v1.reset_default_graph()
        denoise.args = denoise.parser.parse_args([])

        nfilters = nprand.random_integers(4, 128)
        depth = nprand.random_integers(1,7)
        lr = randrange(1e-3, 5e-3)

        memo = "trial{:03d}_nfilters{:d}_depth{:d}_lr{:0.3e}".format(trial, nfilters, depth, lr)
        print("BEGINNING TRIAL: {}".format(memo))
        run_args = [
            'python main_ct.py'
        ]
        denoise.args.phase = 'train'
        denoise.args.gpu = '0,1,2,3'
        denoise.args.weightedloss = True
        denoise.args.geometry = True
        denoise.args.epoch = 40
        denoise.args.traindata_dir = "/home/ryan/projects/MCDoseDenoiser/model_data_2.5mm_crop_rotate"
        denoise.args.log_dir = "/home/ryan/projects/MCDoseDenoiser/logs"
        denoise.args.checkpoint_dir = "/home/ryan/projects/MCDoseDenoiser/checkpoints/{}".format(memo)
        denoise.args.memo = memo

        denoise.args.nfilters = nfilters
        denoise.args.depth = depth
        denoise.args.lr = lr
        denoise.args.batch_size = 100*4
        denoise.args.noeval = False
        denoise.args.nogamma = True

        success = False
        while not success and denoise.args.batch_size>1:
            try:
                denoise.run()
                success = True
            except ResourceExhaustedError as e:
                print(str(e))
                denoise.args.batch_size = math.ceil(denoise.args.batch_size / 2)
                print("reducing batch_size to: {}".format(denoise.args.batch_size))
            except InvalidArgumentError as e:
                print(str(e))
                print("Retrying")
