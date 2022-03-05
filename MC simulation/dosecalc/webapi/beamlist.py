import math
from traceback import format_tb

import payloadtypes
import log
logger = log.get_module_logger(__name__)

def isFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def _parse_line(line):
    beam = payloadtypes.PhotonBeam()
    line = line.replace('\t', ' ')
    tokens = line.split()
    marker = 0
    beam.angle_gantry = float(tokens[0])*math.pi/180.0
    beam.angle_couch  = float(tokens[1])*math.pi/180.0
    marker += 2 # ignore angle_zenith_deg
    if marker < len(tokens) and isFloat(tokens[marker]):
        # get angle_collimator_deg
        beam.angle_coll = float(tokens[2])*math.pi/180.0
        marker += 1
    else:
        beam.angle_coll = 0

    # process optional modules
    while marker < len(tokens):
        if tokens[marker].lower() == 'iso:':
            beam.isocenter = [float(tokens[marker+1])*10.0,
                              float(tokens[marker+2])*10.0,
                              float(tokens[marker+3])*10.0]
            marker += 4
        elif tokens[marker].lower() == 'sad:':
            beam.sad = float(tokens[marker+1])*10.0
            marker += 2
        elif tokens[marker].lower() == 'energy:':
            beam.energy = tokens[marker+1]
            marker += 2
        elif tokens[marker].lower() == 'dir:':
            raise RuntimeError('Setting beam direction vector is not supported')
        elif tokens[marker].lower() == '#':
            # comment for rest of line
            break
        else:
            # unhandled token, skip
            raise RuntimeError('Unknown beam definition module name: "{}"'.format(tokens[marker]))
    return beam

def ensure_exclusive_setting(beams, propname, globalval):
    """require either all beams to have valid property and globalval==None or
                      all beams have None for property and globalval!=None
                      Exception otherwise
    """
    if not beams: return
    beams[0].__dict__[propname] # assert propname is valid key

    if globalval is not None:
        for beam in beams:
            if beam.__dict__.get(propname, None) is not None:
                raise ValueError('Beam property "{:s}" is already set by the global value and cannot be set for individual beams'.format(propname))
            beam.__dict__[propname] = globalval

    else:
        for beam in beams:
            if beam.__dict__.get(propname, None) is None:
                raise ValueError('Beam property "{:s}" must be set for all beams in the beamlist file'.format(propname))


def read_beamlist(f):
    with open(f, 'r') as fd:
        beams = []
        for linenum, line in enumerate(fd):
            line = line.rstrip('\r\n').strip(' ')
            if len(line)<=0 or line[0] == '#':
                logger.debug('skipping comment (line #{:d}): "{!s}"'.format(linenum, line))
                continue
            try:
                beams.append(_parse_line(line))
                logger.debug('parsed line #{:d}: "{!s}"'.format(linenum, line))
            except Exception as err:
                raise RuntimeError('beam definition on line {:d} failed'.format(linenum))
        return beams

