import os
from os.path import join as pjoin
from abc import abstractclassmethod
from math import pi
import copy
import warnings
from collections import namedtuple

from bson import ObjectId
import numpy as np
import scipy.ndimage as ndimage

from api_enums import DBCOLLECTIONS, PROCSTATUS, VARTYPE, STORAGETYPE
import utils
import database
from sparse import load_sparse_data, sparse2dense

class Cache():
    def __init__(self):
        self.cache = {}

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        else:
            val = self.lookup(key)
            self.cache[key] = val
            return val

    @abstractclassmethod
    def lookup(self, key):
        pass
class GeomCache(Cache):
    def lookup(self, key):
        return database.db[DBCOLLECTIONS.MCGEOM].find_one({'_id': key})
class BeamCache(Cache):
    def lookup(self, key):
        return database.db[DBCOLLECTIONS.BEAMPHOTON].find_one({'_id': key})


def rotate_arr(arr, angle, offset=None):
    """rotate array with shape [NDHWC], [NHWC] or [HW] and rotate each 2d slice of shape [HW]
    The resulting shape matches input shape. Angle in radians
    """
    assert arr.ndim in (2, 4, 5)
    if arr.ndim == 2:
        rot_axes = (0, 1)
    elif arr.ndim == 4:
        rot_axes = (1, 2)
    elif arr.ndim == 5:
        rot_axes = (2, 3)

    rotarr = ndimage.rotate(arr, angle=angle*180.0/pi, axes=rot_axes, reshape=False)
    return rotarr

def padcrop_arr(arr, target_height, target_width):
    """Either pad a small centered image with zeros to fill [target_height, target_width]
    or crop large centered image to match [target_height, target_width]
    """
    orig_shape = arr.shape
    assert arr.ndim in (2, 4, 5)

    if arr.ndim == 2:
        h, w = arr.shape[0], arr.shape[1]
    elif arr.ndim == 4:
        h, w  = arr.shape[1], arr.shape[2]
    elif arr.ndim == 5:
        h, w = arr.shape[2], arr.shape[3]

    # pad, does nothing if already large enough
    th = max(h, target_height)
    tw = max(w, target_width)
    oh = max(0, (target_height-h)//2)
    ow = max(0, (target_width-w)//2)

    pad_widths = [(oh, th-oh-h), (ow, tw-ow-w)]
    if arr.ndim == 4:
        pad_widths = [(0,0), *pad_widths, (0,0)]
    elif arr.ndim == 5:
        pad_widths = [(0,0), (0,0), *pad_widths, (0,0)]

    padarr = np.pad(arr, pad_widths, mode='constant')

    # crop, does nothing if already small enough
    oh = max(0, (h-target_height)//2)
    ow = max(0, (w-target_width)//2)

    crop_slices = [slice(oh, oh+target_height), slice(ow, ow+target_width)]
    if arr.ndim == 4:
        crop_slices = [slice(None), *crop_slices, slice(None)]
    elif arr.ndim == 5:
        crop_slices = [slice(None), slice(None), *crop_slices, slice(None)]

    croppadarr = padarr[tuple(crop_slices)]
    return croppadarr


class ArrayLoader():
    interpolation_order = 1
    def __init__(self, reorient=False, context=None, get_label=False, get_geom=False, reversible=False, multiproc=False, dosefactor=1.0, max_samples=None, data_filename='dose3d.bin'):
        """Extract a sample of array data from the database with various post-processing and output options

        Args:
            reorient:   Rotate and center array on beamlet_isocenter to remove beam angle influence.
            context:    (3-int>0) Number of axis-aligned "slices" to include on either side of beamlet central axis.
                        Passing "None" will keep all axis-aligned slices in original array. Note: this will
                        result in cropping along each or one-sided padding if distance to bounds on each side are
                        different and "context" is greater than the distance in the shorter direction.
                        Max value allowed is greater of number of slices on either side of central axis to x-axis bounds
            get_label:  Output ground truth array as well (same post-processing applied)
            get_geom:   Output geometry array as well (same post-processing applied)
            reversible: Build and return UnProcessor object to revert resulting array back to original coordsys
            multiproc:  Force fork-safe access to database (reset db clientsession)
            dosefactor: Immediately after loading the dose array, multiply by factor.
                            Monte Carlo produces dose ~1e-24 so a sane factor would be around 1e24 to prevent
                            datatype truncation errors during pre-processing
        """
        self._geomcache = GeomCache()
        self._beamcache = BeamCache()
        self._reorient = reorient
        self._get_label = get_label
        self._get_geom  = get_geom
        self._reversible = reversible
        self.multiproc = multiproc
        self._context = context
        self._dosefactor = dosefactor
        self.max_samples = max_samples
        self.data_filename = data_filename

    @staticmethod
    def get_beamlet_doc(beamdoc, beamlet_id):
        for beamletdoc in beamdoc['beamlets']:
            if beamletdoc['_id'] == ObjectId(beamlet_id):
                break
            beamletdoc = None
        if beamletdoc is None:
            raise RuntimeError('beamlet doc "{!s}" not found in beamdoc'.format(beamlet_id))
        return beamletdoc

    @staticmethod
    def get_rotmat(radians, axis='z', direction='cw'):
        c, s = np.cos(radians), np.sin(radians)
        if axis == 'z':
            if direction.lower() == 'ccw':
                s *= -1
            elif direction.lower() != 'cw':
                raise ValueError("direction must be either {'cw', 'ccw'}, not '{}'".format(direction.lower()))
            return np.array([[  c,   s, 0.0],
                             [ -s,   c, 0.0],
                             [0.0, 0.0, 1.0]])
        elif axis not in ('x'.lower(), 'y'.lower()):
            raise ValueError("axis must be one of {'x', 'y', 'z'}, not '{}'".format(axis.lower()))
        else:
            raise NotImplementedError()

    def __call__(self, simdoc):
        if self.multiproc:
            # force db reconnect (fork-safety)
            database.reinit_dbclient()

        simdata = self._get_simdata(simdoc)
        coordsys = simdata.coordsys

        if simdoc['storage_type'] == STORAGETYPE.SPARSE:
            def load_bin(path, size):
                return sparse2dense(*load_sparse_data(path))
        elif simdoc['storage_type'] == STORAGETYPE.DENSE:
            def load_bin(path, size):
                return utils.load_bin(path, size)

        arrs = []
        # load inputs
        if simdoc['procstatus']['status'] == PROCSTATUS.SKIPPED:
            zeros_arr = np.zeros(coordsys['size'][::-1], dtype=np.float32)
            nsamples = simdoc['num_runs'] if not self.max_samples else min(simdoc['num_runs'], self.max_samples)
            for _ in range(nsamples):
                arrs.append(zeros_arr)
            if self._get_geom:
                geom_arr  = load_bin(database.dbabspath(
                    simdoc['samples'][0]['densfile']),
                    coordsys['size'])
                arrs.append(geom_arr)
            if self._get_label:
                arrs.append(zeros_arr)
        else:
            nsamples = len(simdoc['samples']) if self.max_samples is None else min(len(simdoc['samples']), self.max_samples)
            for sample in simdoc['samples'][:nsamples]:
                sample_dir = os.path.dirname(database.dbabspath(sample['dosefile']))
                assert sample_dir == database.build_datapath_sample(simdoc['_id'], sample['_id'])
                input_arr = load_bin(pjoin(sample_dir, self.data_filename), coordsys['size'])
                input_arr *= self._dosefactor
                arrs.append(input_arr)

            if self._get_geom:
                geom_arr  = load_bin(database.dbabspath(
                    simdoc['samples'][0]['densfile']),
                    coordsys['size']
                )
                arrs.append(geom_arr)
            if self._get_label:
                label_doc = database.db[DBCOLLECTIONS.SIMULATION].aggregate([
                    {'$match': {
                        'subbeam_id': ObjectId(simdoc['subbeam_id']),
                        'vartype': VARTYPE.LOW,
                        'procstatus.status': PROCSTATUS.FINISHED,
                        'magnetic_field': simdoc['magnetic_field'],
                    }},
                    {'$sort': {'num_particles': -1}} # sort descending
                ]).next()

                label_arr = load_bin(database.dbabspath(
                    label_doc['samples'][0]['dosefile']),
                    coordsys['size']
                )
                label_arr *= self._dosefactor
                arrs.append(label_arr)

        result = self.process(simdoc, arrs, simdata=simdata)
        arrs = result[0]

        # PREPARE OUTPUTS
        outputs = []
        inputs = []
        for ii in range(nsamples):
            inputs.append(arrs[ii])
        outputs.append(inputs)
        for ii in range(nsamples, len(arrs)):
            outputs.append(arrs[ii])

        if self._reversible:
            return outputs, result[1]
        else:
            return (outputs,)

    SimData = namedtuple("SimData", ('coordsys', 'beamlet_iso', 'beamlet_center_fidx', 'beam_angle'))
    def _get_simdata(self, simdoc):
        geomdoc = self._geomcache[ObjectId(simdoc['geom_id'])]
        beamdoc = self._beamcache[ObjectId(simdoc['beam_id'])]
        beamletdoc = self.get_beamlet_doc(beamdoc, simdoc['subbeam_id'])
        coordsys = geomdoc['coordsys']

        # GET DATA FOR POST-PROCESSING
        size = coordsys['size']
        theta = beamdoc['angle_gantry']

        # remember, beamlet['position'] is stored as (pos_z-axis, pos_x-axis)
        beam_center_fidx = (np.array(beamdoc['isocenter']) - np.array(coordsys['start'])) / np.array(coordsys['spacing'])
        rotmat_z_ccw = self.get_rotmat(theta, 'z', 'ccw')
        blt_offset = (
            np.array([beamletdoc['position'][1], 0, beamletdoc['position'][0]]) -
            np.array([beamdoc['fmapdims'][0]   , 0, beamdoc['fmapdims'][1]   ])/2.0 + 0.5
        ) * np.array([beamdoc['beamletsize'][0], 0, beamdoc['beamletsize'][1]])
        beamlet_iso = np.array(beamdoc['isocenter']) + (rotmat_z_ccw @ blt_offset)
        beamlet_center_fidx = (beamlet_iso - np.array(coordsys['start'])) / np.array(coordsys['spacing'])

        simdata = ArrayLoader.SimData(coordsys=coordsys,
                                      beamlet_iso=beamlet_iso,
                                      beamlet_center_fidx=beamlet_center_fidx,
                                      beam_angle=theta)
        return simdata



    def process(self, simdoc, arrs, simdata=None):
        """create an array for a single simulation (possibly multple samples)"""
        if self.multiproc:
            # force db reconnect (fork-safety)
            database.reinit_dbclient()

        if simdata is None:
            simdata = self._get_simdata(simdoc)
        coordsys = simdata.coordsys
        beamlet_iso = simdata.beamlet_iso
        beamlet_center_fidx = simdata.beamlet_center_fidx
        beam_angle = simdata.beam_angle

        if not isinstance(arrs, (list, tuple)):
            arrs = [arrs]

        # 0. SELECT OUTPUTS
        newcoordsys = copy.deepcopy(coordsys)
        orig_size = arrs[0].shape[::-1]

        if self._reorient:
            if self._context is None:
                #if not given, use input array size
                self._context = (np.array(orig_size)-0.5)/2

            # determine y-axis shift/center
            rmat_cw = self.get_rotmat(beam_angle, 'z', 'cw')
            rmat = self.get_rotmat(beam_angle, 'z', 'ccw')
            ix, iy = orig_size[:2]
            in_bounds = np.array([[0,  0, ix, ix],
                                  [0, iy,  0, iy],
                                  [0,  0,  0,  0]])
            out_bounds = rmat_cw @ in_bounds # offsets of corners from top-left in rot. coordsys
            out_bctr = rmat_cw @ beamlet_center_fidx
            rymax, rymin = np.amax(out_bounds[1]), np.amin(out_bounds[1])
            yrange = rymax-rymin
            yshift = 0.5*(rymax+rymin) - out_bctr[1]
            # determine x/z-axes shift/center
            shift = beamlet_center_fidx - (rmat @ (self._context - np.array([0, yshift, 0]) + 0.5))

            out_size = np.ceil(np.array([
                self._context[0]*2+1,
                self._context[1]*2+1,
                self._context[2]*2+1
            ])).astype(int)
            c, s = np.cos(beam_angle), np.sin(beam_angle)
            mat = np.array([[1,  0, 0, shift[2]],
                            [0,  c, s, shift[1]],
                            [0, -s, c, shift[0]]])
            for ii in range(len(arrs)):
                arrs[ii] = ndimage.affine_transform(arrs[ii], mat, order=self.interpolation_order, output_shape=out_size[::-1])
            newcoordsys['start'] = np.array(newcoordsys['start']) + (mat @ np.array([0,0,0,1]))[:3][::-1]*newcoordsys['spacing']
            newcoordsys['size'] = out_size

        if self._reversible:
            reverser = ArrayLoader.UnProcessor(coordsys, newcoordsys, simdata.beam_angle if self._reorient else 0.0)
            return arrs, reverser
        else:
            return (arrs,)


    class UnProcessor():
        def __init__(self, orig_coordsys, proc_coordsys, angle_radian):
            self.orig_csys = orig_coordsys
            self.proc_csys = proc_coordsys
            self.angle_radian = angle_radian

        def __call__(self, arr):
            """implements post-processing 'Undo' to place array back in original coordsys"""
            assert isinstance(arr, np.ndarray)
            # UNSHIFT/ROTATE
            c, s = np.cos(self.angle_radian), np.sin(self.angle_radian)
            offset = (np.array(self.orig_csys['start']) - np.array(self.proc_csys['start']))/np.array(self.orig_csys['spacing'])
            rmat = ArrayLoader.get_rotmat(self.angle_radian, 'z', 'cw')
            shift = rmat @ offset
            mat = np.array([[1,  0,  0, shift[2]],
                            [0,  c, -s, shift[1]],
                            [0,  s,  c, shift[0]]])
            arr = ndimage.affine_transform(arr, mat, order=ArrayLoader.interpolation_order, output_shape=self.orig_csys['size'][::-1])
            return arr
