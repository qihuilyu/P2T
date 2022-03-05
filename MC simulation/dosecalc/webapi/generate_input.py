import sys, os
from os.path import join as pjoin
import math

import numpy as np

from rotation import inv_rotbeam

TEMPLATES = pjoin(os.path.abspath(os.path.dirname(__file__)), 'templates')

def generate_beamon(stream, nparticles, nruns=1):
    """write entire beamon file contents to stream for requested number of particles over requested number of runs"""
    for ii in range(nruns):
        stream.write('/run/beamOn {:d}\n'.format(nparticles))
    return stream

def generate_init(stream, nthreads, magfield=(0,0,1.5, 'tesla'), desc='Geant4 general initialization'):
    with open(pjoin(TEMPLATES, 'init.in.tpl'), 'r') as fd:
        stream.write(
            fd.read().format(
                description=desc,
                nthreads=nthreads,
                magx=magfield[0],
                magy=magfield[1],
                magz=magfield[2],
                magu=magfield[3],
            )
        )
    return stream

def calculate_plane_rotation(angle_gantry, angle_couch, angle_coll):
    """calculate x' and y' unit vectors for given rotations
    # source z' is parallel with beam direction"""
    xp = inv_rotbeam(np.array([1,0,0]), angle_gantry, angle_couch, angle_coll)
    yp = -inv_rotbeam(np.array([0,0,1]), angle_gantry, angle_couch, angle_coll)
    return xp, yp

def is_numeric(x):
    try:
        float(x)
        return True
    except:
        return False

def generate_gps_photon(stream, source, focus, angle_gantry, angle_couch, angle_coll, beamletsize, sad, sfd, energy_mev, desc='Diverging Square field', gps_template=None):
    """Generate the gps input file using a template
    Args:
        idx    (int):          index of beamlet in beam (row-major order)
        source (x, y, z, u):   coordinates
        focus  (x, y, z, u):   coordinates
        angle_gantry  (float): gantry angle
        beamletsize (x, z, u)
        sad    (float):        sad (units must match beamletsize units)
        sfd    (float):        src-focus-distance (units must match beamletsize units)
    """
    extra_kwargs = {}
    # try to match requested template
    if gps_template is not None:
        fullpath = pjoin(TEMPLATES, gps_template)
        if not os.path.isfile(fullpath):
            raise FileNotFoundError('GPS template "{}" doesn\'t exist'.format(fullpath))
    else:
        if energy_mev is not None and is_numeric(energy_mev):
            gps_template = 'gps_photon_mono.mac.tpl'
            extra_kwargs['energy'] = float(energy_mev)
        else:
            gps_template = 'gps_photon_6MV.mac.tpl'

    xp, yp = calculate_plane_rotation(angle_gantry, angle_couch, angle_coll)
    adj_fsize = [0.5*sfd/sad*beamletsize[ii] for ii in range(2)]
    with open(pjoin(TEMPLATES, gps_template), 'r') as fd:
        stream.write(
            fd.read().format(
                description=desc,
                cx=source[0],
                cy=source[1],
                cz=source[2],
                cu=source[3],
                rot1x=xp[0],
                rot1y=xp[1],
                rot1z=xp[2],
                rot2x=yp[0],
                rot2y=yp[1],
                rot2z=yp[2],
                fsx=adj_fsize[0],
                fsy=adj_fsize[1],
                fsu=beamletsize[2],
                fx=focus[0],
                fy=focus[1],
                fz=focus[2],
                fu=focus[3],
                **extra_kwargs,
            )
        )
    return stream

def generate_gps_electron(stream, source, focus, angle_gantry, angle_couch, angle_coll, beamletsize, sad, sfd, energy_mev, desc='Diverging Square field', gps_template=None):
    """Generate the gps input file using a template
    Args:
        idx    (int):           index of beamlet in beam (row-major order)
        source (x, y, z, u):    coordinates
        focus  (x, y, z, u):    coordinates
        angle_gantry  (float):  gantry angle
        beamletsize (x, z, u)
        sad    (float):         sad (units must match beamletsize units)
        sfd    (float):         src-focus-distance (units must match beamletsize units)
        energy (float):         Mono-energetic beam energy (units: MeV)
    """
    # try to match requested template
    if gps_template is not None:
        fullpath = pjoin(TEMPLATES, gps_template)
        if not os.path.isfile(fullpath):
            raise FileNotFoundError('GPS template "{}" doesn\'t exist'.format(fullpath))
    else:
        gps_template = 'gps_electron_mono.mac.tpl'

    xp, yp = calculate_plane_rotation(angle_gantry, angle_couch, angle_coll)
    adj_fsize = [0.5*sfd/sad*beamletsize[ii] for ii in range(2)]
    with open(pjoin(TEMPLATES, gps_template), 'r') as fd:
        stream.write(
            fd.read().format(
                description=desc,
                cx=source[0],
                cy=source[1],
                cz=source[2],
                cu=source[3],
                rot1x=xp[0],
                rot1y=xp[1],
                rot1z=xp[2],
                rot2x=yp[0],
                rot2y=yp[1],
                rot2z=yp[2],
                fsx=adj_fsize[0],
                fsy=adj_fsize[1],
                fsu=beamletsize[2],
                fx=focus[0],
                fy=focus[1],
                fz=focus[2],
                fu=focus[3],
                energy=float(energy_mev),
            )
        )
    return stream

def generate_gps_proton(stream, source, focus, angle_gantry, angle_couch, angle_coll, beamletsize, sad, sfd, energy_mev, desc='Diverging Square field', gps_template=None):
    """Generate the gps input file using a template
    Args:
        idx    (int):           index of beamlet in beam (row-major order)
        source (x, y, z, u):    coordinates
        focus  (x, y, z, u):    coordinates
        angle_gantry  (float):  gantry angle
        beamletsize (x, z, u)
        sad    (float):         sad (units must match beamletsize units)
        sfd    (float):         src-focus-distance (units must match beamletsize units)
        energy (float):         Mono-energetic beam energy (units: MeV)
    """
    # try to match requested template
    if gps_template is not None:
        fullpath = pjoin(TEMPLATES, gps_template)
        if not os.path.isfile(fullpath):
            raise FileNotFoundError('GPS template "{}" doesn\'t exist'.format(fullpath))
    else:
        gps_template = 'gps_proton_mono.mac.tpl'

    xp, yp = calculate_plane_rotation(angle_gantry, angle_couch, angle_coll)
    adj_fsize = [0.5*sfd/sad*beamletsize[ii] for ii in range(2)]
    with open(pjoin(TEMPLATES, gps_template), 'r') as fd:
        stream.write(
            fd.read().format(
                description=desc,
                cx=source[0],
                cy=source[1],
                cz=source[2],
                cu=source[3],
                rot1x=xp[0],
                rot1y=xp[1],
                rot1z=xp[2],
                rot2x=yp[0],
                rot2y=yp[1],
                rot2z=yp[2],
                fsx=adj_fsize[0],
                fsy=adj_fsize[1],
                fsu=beamletsize[2],
                fx=focus[0],
                fy=focus[1],
                fz=focus[2],
                fu=focus[3],
                energy=float(energy_mev),
            )
        )
    return stream
