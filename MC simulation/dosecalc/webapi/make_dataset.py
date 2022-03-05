#!/usr/bin/env python
"""process monte carlo simulated data into numpy arrays for use during network training
Data samples (slices) for each patient will be saved as a concatenated tensor to a single common .npy file which can be used to construct batches during training
"""

import os, sys
from os.path import join as pjoin
import numpy as np
import json
import math
import logging
from scipy.ndimage import interpolation
from tqdm import tqdm

from utils import load_bin

norm_factor = 2.65e-25

def rotate(arr, theta):
    """rotate array 'theta' radians clockwise"""
    return interpolation.rotate(arr, theta*180.0/math.pi, axes=(1,0), reshape=False, order=2, mode='constant', cval=np.min(arr))

def run(sampleids, dest_dir, mode='train'):
    # give a list of sample ids to include in dataset ( must be for same CT )
    # produces set of files:
    # - angles_train.py
    # - geo_train.py
    # - high_train.py
    # - low_train.py
    # - sampleids_train.py

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler('error_log.txt'))

    doi_name = os.path.basename(doi)
    results_dir = pjoin(dest_dir, doi_name)
    f_high   = pjoin(results_dir, 'high_{}.npy'.format(mode))
    f_low    = pjoin(results_dir, 'low_{}.npy'.format(mode))
    f_geo    = pjoin(results_dir, 'geo_{}.npy'.format(mode))
    f_angles = pjoin(results_dir, 'angles_{}.npy'.format(mode))

    if True:
        #  if all([os.path.isfile(x) for x in (f_high, f_low, f_geo)]) and not args.overwrite:
        #      print('Skipping model data construction for {} ({})'.format(doi_name, mode))
        #      continue

        print('Constructing model data for {} ({})'.format(doi_name, mode))
        dose_h = []
        dose_l = []
        geom = []
        angles = []

        print('  +loading simulation data')
        for blt in tqdm(get_beamlet_dirs(doi)):
            blt_low = pjoin(src_dir_low, blt.replace(src_dir, src_dir_low))
            with open(pjoin(blt, 'config.json'), 'r') as fp:
                config = json.load(fp)
                sliceno = config['sub_sliceno']
                size = config['sub_size']
                try:
                    sub_crop_start = config['sub_crop_start']
                    sub_crop_size = config['sub_crop_size']
                    theta = config['theta']
                    crop_slice = [slice(sub_crop_start[1-ii], sub_crop_start[1-ii]+sub_crop_size[1-ii]) for ii in range(2)]
                except:
                    print("Cannot crop because crop params are not present in config")
                    args.crop = False

            try:
                arr_low = load_bin(pjoin(blt_low, args.results_pattern, 'lowvar.in_0', 'dose3d.bin'), size)[sliceno].astype(np.float32)
                arr_geo = load_bin(pjoin(blt, args.results_pattern, 'highvar.in_0', 'InputDensity.bin'), size)[sliceno].astype(np.float32)
                if args.crop:
                    arr_low = arr_low[crop_slice]
                    arr_geo = arr_geo[crop_slice]
                if args.rotate:
                    arr_low = rotate(arr_low, theta)
                    arr_geo = rotate(arr_geo, theta)

                for sim in get_sim_results(pjoin(blt, args.results_pattern)):
                    arr = load_bin(pjoin(sim, 'dose3d.bin'), size)[sliceno].astype(np.float32)
                    if args.crop:
                        arr = arr[crop_slice]
                    if args.rotate:
                        arr = rotate(arr, theta)
                    dose_h.append(arr)

                    # reuse the same low var dose
                    dose_l.append(arr_low)
                    geom.append(arr_geo)
                    angles.append(theta)
                    for a in [arr_geo, arr]:
                        assert a.shape == arr_low.shape
            except FileNotFoundError as e:
                logger.warning(str(e))


        # concat
        print('  +combining arrays and saving to file')
        full_arr_high = np.stack(dose_h)/norm_factor
        full_arr_low = np.stack(dose_l)/norm_factor
        full_arr_geo = np.stack(geom)
        full_angles = np.array(angles)

        def print_stats(arr):
            print('    shape: ({:d},{:d},{:d})'.format(*arr.shape))
            print('    min:   {:e}'.format(np.min(arr)))
            print('    max:   {:e}'.format(np.max(arr)))
            print('    mean:  {:e}'.format(np.mean(arr)))

        print('Data Statistics:')
        print('  low variance:')
        print_stats(full_arr_low)
        print('  high variance:')
        print_stats(full_arr_high)

        os.makedirs(results_dir, exist_ok=True)
        np.save(f_high, full_arr_high)
        np.save(f_low, full_arr_low)
        np.save(f_geo, full_arr_geo)
        np.save(f_angles, full_angles)
        print()
        logger.info('\n')
