import math
from subprocess import run, CalledProcessError

import numpy as np
import numpy.random as nprand


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=1, help='first trial index for logging')
parser.add_argument('--trials', type=int, default=50, help='number of trials to run')


def randrange(start, stop, size=None):
    return (stop-start)*nprand.random_sample(size) + start

if __name__ == "__main__":
    args = parser.parse_args()

    for trial in range(args.start, args.start+args.trials+1):
        nfilters = nprand.random_integers(4, 128)
        depth = nprand.random_integers(1,7)
        lr = randrange(1e-3, 5e-3)

        memo = "trial{:03d}_nfilters{:d}_depth{:d}_lr{:0.3e}".format(trial, nfilters, depth, lr)
        print("BEGINNING TRIAL: {}".format(memo))
        run_args = [
            'python', 'main_ct.py',
            '--gpu', '0,1,2,3',
            '--weightedloss',
            '--geometry',
            '--epoch', '40',
            '--phase', 'train',
            '--traindata_dir', '/home/ryan/projects/MCDoseDenoiser/model_data_2.5mm_crop_rotate',
            '--log_dir', '/home/ryan/projects/MCDoseDenoiser/logs',
            '--checkpoint_dir', "/home/ryan/projects/MCDoseDenoiser/checkpoints/{}".format(memo),
            '--memo', str(memo),

            '--nfilters', str(nfilters),
            '--depth', str(depth),
            '--lr', str(lr),
            '--nogamma',
        ]

        success = False
        batch_size = 100*4
        while not success and batch_size>1:
            try:
                res = run(run_args+['--batch_size', str(batch_size)], check=True, encoding='utf-8')
                success = True
            except CalledProcessError as e:
                print(str(e))
                batch_size = math.ceil(batch_size / 2)
                print("reducing batch_size to: {}".format(batch_size))
                print("Retrying")
