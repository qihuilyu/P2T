from multiprocessing import cpu_count

import numpy as np

from ._low_level import calc_gamma

def gamma_passing_rate(gamma_map, mask=None):
    """calculate gamma passing rate as percentage of voxels with gamma index >= 1.0
    Optionally, a mask may be provided which excludes all voxels with mask==0"""
    if mask is None:
        passing = np.count_nonzero(gamma_map<=1.0)
        total = gamma_map.size
    else:
        mask = mask.astype(bool)
        masked_gamma_map = gamma_map[mask]
        try:
            passing = np.count_nonzero(masked_gamma_map<=1.0)
            total = masked_gamma_map.size
        except:
            passing = 0
            total = 0
            logger.warning('Mask contained no Truth values, passing rate cannot be calculated')
    return passing, total

def gamma_analysis(low_var, output_low_var, dd, dta, voxelsize, num_threads=4, output_components=False):
    """gamma analysis (note: dta is in mm, dd in fraction of max; as in 2mm/2%)"""
    num_threads = min(num_threads, cpu_count())
    num_threads = 1 # force single-threaded; is faster for this data
    coords = tuple([np.arange(0, low_var.shape[ii]*voxelsize, voxelsize) for ii in range(low_var.ndim)])
    res = calc_gamma(
        coords, low_var, coords, output_low_var,
        dta, dd, maximum_test_distance=3*dta,
        lower_dose_cutoff=-np.inf,
        num_threads=num_threads, output_components=output_components)
    if output_components:
        gamma_map, dd_map, dta_map = res
    else:
        gamma_map = res
    gamma_map[np.isnan(gamma_map)] = np.inf

    if output_components:
        return gamma_map, dd_map, dta_map
    else:
        return gamma_map
