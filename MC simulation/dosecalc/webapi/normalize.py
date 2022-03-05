import sys, os
import shutil
from os.path import join as pjoin
import json
import argparse
import logging
import multiprocessing

from tqdm import tqdm
import numpy as np

logger = logging.getLogger()

class StatsRecorder:
    def __init__(self):
        """
        data: tensor [C, N*D*H*W]
        """
        self.nobservations = 0
        self.mean = 0
        self.std = 0

    def update(self, arr):
        newmean = np.mean(arr, axis=1)
        newstd = np.std(arr, axis=1)

        m = float(self.nobservations)
        n = float(arr.shape[1])

        tmp = self.mean

        self.mean = m/(m+n)*tmp + n/(m+n)*newmean
        self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                    m*n/(m+n)**2 * (tmp - newmean)**2
        self.std  = np.sqrt(self.std)

        self.nobservations += n


def load_np_calcstats(f):
    arr = np.load(f)
    # loads with shape [N,D,H,W,C]
    arr = np.transpose(arr, (4,0,1,2,3)) # make channel axis first ([C,N,D,H,W])
    arr = np.reshape(arr, (arr.shape[0], -1)) # flatten [N,D,H,W] into single axis [C,N*D*H*W]
    return arr

def load_np_update(f, mean_arr, std_arr):
    arr = np.load(f)
    arr = (arr-mean_arr) / std_arr
    arr = arr.astype(np.float32)
    return arr

def normalize_by_zscore(root, out, files):
    statsrecord = StatsRecorder()
    for f in tqdm(files, desc='Calc. stats'):
        arr = load_np_calcstats(f)
        statsrecord.update(arr)

    mean_arr = statsrecord.mean
    std_arr = statsrecord.std
    stats = {
        'mean': statsrecord.mean.tolist(),
        'std':  statsrecord.std.tolist(),
    }

    # only normalize by ground truth statistics [GT, IN, GEOM]
    mean_arr = np.stack([mean_arr[0], mean_arr[0], mean_arr[2]])[None,None,None,None,:]
    std_arr  = np.stack([std_arr[0],  std_arr[0],  std_arr[2] ])[None,None,None,None,:]

    for f in tqdm(files, desc='Update files'):
        # apply norm. to all files
        arr = load_np_update(f, mean_arr, std_arr)
        fout = f.replace(root, out)
        os.makedirs(os.path.dirname(fout), exist_ok=True)
        np.save(fout, arr)

    return stats


def normalize_dataset(root, out):
    # Network auto-applies normalization at test time so only train/validation data needs to be adjusted
    search_dirs = [pjoin(root, x) for x in ('train', 'validate')]
    files = []
    for d in search_dirs:
        files += get_npy_files(d)

    os.makedirs(out, exist_ok=True)
    normstats = normalize_by_zscore(root, out, files)
    with open(pjoin(root, 'stats.json'), 'r') as fd:
        jstats = json.load(fd)
    jstats = {**jstats, **normstats}
    with open(pjoin(out, 'stats.json'), 'w') as fd:
        json.dump(jstats, fd)

    aux_files = get_aux_files(root)
    for f in aux_files:
        fout = f.replace(root, out)
        shutil.copy2(f, fout)

def get_files(folder, recursive, ext=None):
    if not isinstance(ext, (list, tuple)):
        ext = [ext]
    paths = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if ext and os.path.splitext(f)[1].lower() in ext:
                paths.append(pjoin(root, f))
        if not recursive:
            del dirs[:]
    return paths

def get_npy_files(folder, recursive=True):
    return get_files(folder, recursive, ext='.npy')
def get_aux_files(folder, recursive=True):
    return get_files(folder, recursive, ext=['.txt'])

if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('root', help='directories to search for files')
    parser.add_argument('out', help='output directory for standardized data (copy)')
    parser.add_argument('--type', choices=['zscore'], default='zscore', help='which type of normalization to apply')
    args = parser.parse_args()
    normalize_dataset(args.root, args.out)

