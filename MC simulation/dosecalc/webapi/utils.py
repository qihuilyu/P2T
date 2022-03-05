import os
import shlex
import struct

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from payloadtypes import CoordSys

def split_string(s):
    """split string to list"""
    if s is None:
        return []
    else:
        return shlex.split(s)


def get_directory_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def load_bin(fname, size, add_channel_axis=False):
    """Read floating-point bytes and create numpy array"""
    with open(fname, 'rb') as fd:
        databuf = fd.read()
    except_msgs = []
    for type in ['f', 'd']:
        try:
            arr = np.array(struct.unpack(type*np.product(size), databuf)).reshape(size[::-1])
            break
        except Exception as e:
            except_msgs.append(str(e))
            continue
        if arr is None:
            raise Exception("\n".join(except_msgs))
    if add_channel_axis:
        arr = np.expand_dims(arr, axis=arr.ndim)
    return arr

def save_bin(fname, arr):
    arr.astype(np.float32).tofile(fname)

def get_resizing_params(full_coordsys, sub_coordsys):
    assert full_coordsys['start'] <= sub_coordsys['start']
    assert full_coordsys['size'] >= sub_coordsys['size']
    assert full_coordsys['spacing'] == sub_coordsys['spacing']
    offset = np.floor((np.subtract(sub_coordsys['start'], full_coordsys['start'])) / full_coordsys['spacing']).astype(int)
    subslice = tuple([slice(offset[ii], offset[ii]+sub_coordsys['size'][ii]) for ii in range(3)])[::-1]
    return subslice

def resample(arr, source_grid, target_grid):
    """resample array from source_grid to target_grid

    Args:
        arr: 3D np.array containing data
        source_grid: CoordSys defining original grid
        target_grid: CoordSys defining target grid
    """
    if isinstance(source_grid, dict):
        source_grid = CoordSys(**source_grid)
    if isinstance(target_grid, dict):
        target_grid = CoordSys(**target_grid)
    assert isinstance(source_grid, CoordSys)
    assert isinstance(target_grid, CoordSys)
    print(source_grid, target_grid)


    # define interpolator
    scx = source_grid.start[0] + np.arange(source_grid.size[0])*source_grid.spacing[0]
    scy = source_grid.start[1] + np.arange(source_grid.size[1])*source_grid.spacing[1]
    scz = source_grid.start[2] + np.arange(source_grid.size[2])*source_grid.spacing[2]
    interp = RegularGridInterpolator((scz, scy, scx), arr, bounds_error=False, fill_value=0, method='linear')

    # evaluate interpolator
    cx, cy, cz = np.meshgrid(np.arange(target_grid.size[0]), np.arange(target_grid.size[1]), np.arange(target_grid.size[2]))
    tcx = target_grid.start[0] + cx*target_grid.spacing[0]
    tcy = target_grid.start[1] + cy*target_grid.spacing[1]
    tcz = target_grid.start[2] + cz*target_grid.spacing[2]
    target_coords = np.stack([tcz, tcy, tcx], axis=-1)
    print(target_coords.shape)
    resampled_arr = interp(target_coords)
    return resampled_arr

def none_or_type(type):
    def f(v):
        if isinstance(v, str) and v.lower() == 'none':
            return None
        else:
            return type(v)
    return f
