import sys, os
from os.path import join as pjoin
import io
import shutil
import math
from math import cos, sin, floor
import numpy as np
import scipy.io as sio
from raytrace import spottrace, beamtrace

from rotation import inv_rotbeam, rot_around_axis_rhs
from ct2mat import lookup_materials
import log

logger = log.get_module_logger(__name__)

MACHINE_DATA = pjoin(os.path.abspath(os.path.dirname(__file__)), 'machine_data')

def get_active_spots(density, mask, angle_gantry, angle_couch, angle_coll, iso, start, spacing, fmapdims, beamletspacing, beamletsize, sad):
    """Produce a list of spot"""
    depths = spottrace(
            sad=sad,
            det_dims=fmapdims,
            det_center=iso,
            det_spacing=beamletspacing,
            det_pixelsize=beamletsize,
            det_azi=angle_gantry,
            det_zen=angle_couch,
            det_ang=angle_coll,
            vols=[density, mask],
            vol_start=start,
            vol_spacing=spacing
            )

    positions = []
    for iz in range(fmapdims[1]):
        for ix in range(fmapdims[0]):
            segments = depths[iz, ix]
            if len(segments) <= 0:
                continue
            energies = rad_depths_to_spot_energies(segments)
            # segments = [(min_depth, max_depth), (min_depth, max_depth)] = depths[iz, ix]
            # segments = [(d1,d2), (d1,d2), (d1,d2)]
            # energies = [[1, 2, 3], [6,7], [8]]
            # flat_energies = [1,2,3,6,7,8]
            flat_energies = []
            for segment_energies in energies:
                flat_energies += segment_energies
            positions += list(zip([(iz, ix)]*len(flat_energies), flat_energies))
    return positions

def get_active_beamlets(mask, angle_gantry, angle_couch, angle_coll, iso, start, spacing, fmapdims, beamletspacing, beamletsize, sad, vispath=None):
    """call ray tracing code to determine which beamlets pass through PTV for a given isocenter, beam angle,
    and beamlet position. Treat all coordinates as dimensionless (indices)
    Args:
        mask (nparray):          1 inside, 0 outside
        angle_gantry (f):        beam gantry angle in radians (0 rad beam shoots in +y direction; coplanar angle)
        angle_couch (f):         beam couch angle in radians
        angle_coll (f):          collimator rotation angle in radians
        iso (fx, fy, fz):        coordinates of isocenter
        start (fx, fy, fz):      coordinates of first pixel in mask array
        spacing (fx, fy, fz):    spacing of mask array pixels (voxelsize)
        fmapdims (ix, iz):       dimensions of square beam (number of beamlets along each axis)
        beamletspacing (fx, fz): distance between adjacent beamlet centers
        beamletsize (fx, fz):    size of beamlet at isocenter
        sad (f):                 source-to-isocenter distance
    """
    assert mask.ndim == 3
    assert len(iso) == 3
    #  fmap_hits = raytrace(dests, sources, mask, start, spacing, stop_early=-1 if vispath else -1).reshape(fmapdims)
    fmap_hits = beamtrace(sad, fmapdims, iso, beamletspacing, beamletsize, angle_gantry, angle_couch, angle_coll, mask, start, spacing, stop_early=9 if vispath else 9).reshape(fmapdims[::-1])

    if vispath:
        # this is done implicitly within beamtrace but we do it explicitly for vis/debug purposes
        # TODO: Update this for non-coplanar visualization
        rot_offset = np.array([[cos(angle_gantry), 0, 0, (iso[0])],
                               [sin(angle_gantry), 0, 0, (iso[1])],
                               [0,          0, 1, (iso[2])]])
        xx, yy = np.meshgrid(np.arange(fmapdims[0]), np.arange(fmapdims[1]))
        xx = xx.ravel()
        yy = yy.ravel()
        bev_indices = np.vstack(((xx-0.5*(fmapdims[0]-1))*beamletspacing[0],
                                 np.zeros_like(xx),
                                 (yy-0.5*(fmapdims[1]-1))*beamletspacing[1],
                                 np.ones_like(xx)
                                 ))
        dests = np.dot(bev_indices.T, rot_offset.T)
        source = (sad*sin(angle_gantry) + iso[0], -sad*cos(angle_gantry) + iso[1], iso[2])
        dests = 2*dests - source # extend beyond isocenter for full raytracing

        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1,3,1)
        image_extent = [start[0], start[0]+mask.shape[2]*spacing[0],
                        start[1]+mask.shape[1]*spacing[1], start[1]]
        plt.imshow(mask[int(floor((iso[2]-start[2])/spacing[2])), :, :], extent=image_extent)
        plt.plot([source[0], iso[0]], [source[1], iso[1]], c='blue', zorder=2)
        plt.scatter(source[0], source[1], c='blue', zorder=2)
        plt.scatter(iso[0], iso[1], c='red', zorder=2)
        plt.scatter([d[0] for d in dests], [d[1] for d in dests], c='green', zorder=2)
        for d in (dests[0], dests[-1]):
            plt.plot([source[0], d[0]], [source[1], d[1]], linewidth=0.5, c='gray', zorder=1)
        plt.subplot(1,3,2)
        plt.imshow(fmap_hits, origin='lower')
        plt.subplot(1,3,3)
        plt.imshow(fmap_hits>0, origin='lower')
        plt.tight_layout()
        plt.savefig(pjoin(vispath, 'beamtrace.png'))
        #  plt.show()

    return np.argwhere(fmap_hits>0).tolist(), fmap_hits

def calculate_gps_coordinates(position, angle_gantry, angle_couch, angle_coll, iso, start, size, spacing, fmapdims, beamletspacing, beamletsize, sad):
    """Calculate source and focus points (all units in mm)
    The (0,0)-degree beam is in the +y direction
        (90,0)-degree beam is in the -x direction
        (90,90)-degree beam is in the -z direction
    Collimator rotation follows Right-hand rule with scanner +y axis
    fluence map +x/+y axes correspond to scanner +x/+z axes

    Geant4 centers the array at the origin so source and iso need to be shifted accordingly
    the orientation also changes, so the typical z-perpendicular coplanar surface is actually

    Args:
        position: (z, x) index of beamlet in beam
        angles: ...
        iso (x, y, z): in millimeters, scanner coordinates
        start (x, y, z): in millimeters, scanner coords
        size
        spacing (x, y, z): in mm
    """
    iso = np.array(iso)
    start = np.array(start)
    size = np.array(size)
    spacing = np.array(spacing)
    fmapdims = np.array(fmapdims)
    beamletsize = np.array(beamletsize)
    beamletspacing = np.array(beamletspacing)
    position = np.array(position[::-1]) #position is normally (y, x) so this puts it in (x, y) order

    #=========================
    # non-coplanar positioning
    #=========================
    sfd = sad/10.0
    # TODO: REMOVE SHIFTED START/ISO (from geant4 as well) IN NEXT MAJOR RELEASE (breaks compatability)
    shift = 0.5*np.multiply(size-1, spacing)
    iso = np.subtract(np.subtract(iso, start), shift)
    mag_factor = (sfd/sad)
    # setup source plane center at sad+sfd, and scale beamlet offset according to magnification
    mag_offset = np.array([[-mag_factor, 0,           0,          0],
                           [          0, 0,           0, -(sfd+sad)],
                           [          0, 0, -mag_factor,          0]])
    fmap_pos = np.array([(position[0] - (fmapdims[0]-1)/2.0)*beamletspacing[0],
                         0,
                         (position[1] - (fmapdims[1]-1)/2.0)*beamletspacing[1],
                         1 ])
    src_ = np.dot(mag_offset, fmap_pos)
    source = inv_rotbeam(src_,                   angle_gantry, angle_couch, angle_coll) + iso
    focus  = inv_rotbeam(np.array([0, -sad, 0]), angle_gantry, angle_couch, angle_coll) + iso
    return source, focus, iso

def generate_geometry(f_geo, vol, voxelsize, bulk_density=False, isoshift=np.array([0,0,0])): # QL
    """Atomic method for producing MC geometry files"""
    # ZYX ORDERING
    # note: transpose volume if you need to change iteration order

    materials = lookup_materials(ctnums=vol)

    buf = io.StringIO()
    for chunk in np.nditer(materials, flags=['external_loop', 'buffered', 'refs_ok'], op_flags=['readonly'], order="C"):
        for mat_def in chunk:
            buf.write(mat_def.get_material_def())

    with open(f_geo, 'w') as fd:
        fd.write("{:d} {:d} {:d}\n".format(*vol.shape[::-1]))
        fd.write("{:f} {:f} {:f}\n".format(*voxelsize))
        fd.write("{:f} {:f} {:f}\n".format(isoshift[0], isoshift[1], isoshift[2]))  # QL
        buf.seek(0)
        shutil.copyfileobj(buf, fd)

# input: list of of radiological depths pairs (D_min,D_max) for each beamlet
# output: list of spot energies that cover that range of depths
def rad_depths_to_spot_energies(beamlet_depth_ranges):
    """use for each beamlet individually. beamlet_depth_ranges should be a list of 2-tuples for a single beamlet, with each 2-tuple describing
    (min_depth, max_depth) for a segment of intersection with the target structure

    beamlet_depth_ranges = [(min_depth_1, max_depth_1), (min_depth_2, max_depth_2), ...]
    """
    # Use machine data from protons_Generic.mat
    table = sio.loadmat(pjoin(MACHINE_DATA, 'protons_Generic.mat') ,squeeze_me=True)
    data = table['machine']['data'].item()
    machineEnergies = data['energy']
    depths = data['peakPos'] + data['offset']
    # For each beamlet, find the spot energies that cover the depth range
    beamlet_energies = [None]*len(beamlet_depth_ranges)
    for idx, (D_min, D_max) in enumerate(beamlet_depth_ranges):
        energies = []
        for i in range(len(depths)):
            if D_min <= depths[i] <= D_max:
                energies.append(machineEnergies[i])
        beamlet_energies[idx] = energies
    return beamlet_energies
