import sys, os
import re
import pydicom
import dicom_numpy
from math import cos, sin, floor, pi

import numpy as np
import numpy.random as rand
from rttypes.dcmio import CTIMAGE_SOP_CLASS_UID, MRIMAGE_SOP_CLASS_UID, PETIMAGE_SOP_CLASS_UID, RTIMAGE_SOP_CLASS_UID, RTSTRUCT_SOP_CLASS_UID
from rttypes.roi import ROI
import log
logger = log.get_module_logger(__name__)

modalities = {CTIMAGE_SOP_CLASS_UID: "CT",
              MRIMAGE_SOP_CLASS_UID: "MR",
              PETIMAGE_SOP_CLASS_UID: "PET",
              RTIMAGE_SOP_CLASS_UID:  "RTIMAGE",
              RTSTRUCT_SOP_CLASS_UID: "RTSTRUCT"}

def find_dicom_files(root, recursive=False):
    """search (recursively) for dicom files and return a dictionary of files organized by modality"""
    reg = re.compile(r"\.(?:dcm|dicom)")
    dicom_files = {}
    for base, dirs, files in os.walk(root):
        for f in files:
            filename = os.path.join(base, f)
            if re.search(reg,filename.lower()):
                with open(filename,'rb') as fd:
                    # organize by modality
                    modality = modalities[pydicom.dcmread(fd, stop_before_pixels=True).SOPClassUID]
                    if not modality in dicom_files:
                        dicom_files[modality] = []
                    dicom_files[modality].append(filename)
        if not recursive:
            del dirs[:]
    return dicom_files

def get_dicom_seriesuid(dicomfile):
    with open(dicomfile, 'rb') as fd:
        return pydicom.dcmread(fd, stop_before_pixels=True).SeriesInstanceUID

def get_dicom_voxelsize(dicomfile):
    with open(dicomfile, 'rb') as fd:
        data = pydicom.dcmread(fd, stop_before_pixels=True)
        return (*data.PixelSpacing, data.SliceThickness)

def _find_mim_dir(path, typestr):
    for d in [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]:
        m = re.search(r'(?:.*)_(?:.*)_([a-zA-Z ]+)_.*', d, re.IGNORECASE)
        if m is None:
            continue
        mod = m.group(1).lower()

        if mod.lower() == typestr.lower():
            return d

def find_dcm_dir(path):
    return _find_mim_dir(path, 'ct')

def find_rtstr_file(path):
    rtstr_dir = _find_mim_dir(path, 'rtst')
    if rtstr_dir is None: return None
    rtfilepath = os.path.join(rtstr_dir, [x for x in os.listdir(os.path.join(path, rtstr_dir)) if os.path.splitext(x)[1].lower() in ['.dcm', '.dicom']][0])
    return rtfilepath

def get_roi_names(f_rtstr):
    """returns list of structures in the RTStruct file"""
    return ROI.getROINames(f_rtstr)

def generate_mask(f_rtstr, frame, ptv_name, margin=None):
    """generate ROI binary volume/mask from rtstruct file and frameofreference from dicom volume"""
    roi = ROI.roiFromFile(f_rtstr, ptv_name)
    if roi is None:
        raise RuntimeError('Roi "{}" could not be located in "{}"'.format(ptv_name, f_rtstr))
    try:
        mask = roi.makeDenseMask(frame, margin=margin).data.astype(np.int8)
    except TypeError:
        logger.warning('Failed to make mask due to empty rtstruct coordinates list')
    return mask

def validate_bbox(bbox, frame):
    """ensure bbox is completely contained within frame clip bbox to frame if necessary"""
    bbox.start = tuple([min(frame.end()[ii]-frame.spacing[ii], max(frame.start[ii], bbox.start[ii])) for ii in range(3)])
    bbox.size = tuple([int(min((frame.end()[ii]-bbox.start[ii])/bbox.spacing[ii], max(1, bbox.size[ii]))) for ii in range(3)])
    return bbox

def get_roi_bbox(f_rtstr, frame, roi_name, buffer=0):
    """get tight coordinate system around structure, added optional buffer (in physical units) to each side,
    then validate against full coordinate system, clipping when necessary"""
    roi = ROI.roiFromFile(f_rtstr, roi_name, casesensitive=False)
    if roi is None:
        raise RuntimeError('Roi "{}" could not be located in "{}"'.format(roi_name, f_rtstr))
    extents = roi.getROIExtents(spacing=frame.spacing)
    extents.spacing = frame.spacing # in case this isn't explicitly set by rttypes library
    # add buffer to each side
    extents.start = np.subtract(extents.start, buffer)
    extents.size = np.add(extents.size, np.divide(2*buffer, frame.spacing))
    return validate_bbox(extents, frame)

def centroid_as_coords(mask, frame):
    """return center of mass as coordinates with respect to dicom frame of reference in xyz order"""
    centroid_indices = np.average(np.argwhere(mask), axis=0).tolist()
    return (np.array(centroid_indices[::-1])*frame.spacing+frame.start).tolist()

def generate_spaced_beams(n, start=None):
    beams = []
    if start is None:
        beam_ang_start = rand.uniform(0, 2.0*pi)
    elif isinstance(start, (tuple, list)):
        assert len(start) == 2
        s, e = sorted(start)
        beam_ang_start = rand.uniform(s, e)
    else:
        beam_ang_start = float(start)

    beam_ang_spacing = 2.0*pi/n
    for aa in range(n):
        theta = (beam_ang_start + beam_ang_spacing*aa) % (2*pi)
        beams.append(theta)
    return beams

def extract_voxel_data(list_of_dicom_files):
    datasets = [pydicom.read_file(f) for f in list_of_dicom_files]
    try:
        voxel_ndarray, affine = dicom_numpy.combine_slices(datasets)
        voxelsize = (affine[0,0].item(), affine[1,1].item(), affine[2,2].item())
        # rotate to restore dicom coordinate system alignment
        voxel_ndarray = voxel_ndarray.transpose(2,1,0).copy("C")
    except dicom_numpy.DicomImportException as e:
        raise # invalid DICOM data
    return voxel_ndarray, voxelsize
