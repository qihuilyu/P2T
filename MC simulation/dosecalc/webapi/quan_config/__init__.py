import os
from os.path import join as pjoin
import math
from typing import List, Tuple

class ControlPoint():
    def __init__(self):
        self.leaf_edges = [] # leaf positions at isocenter [1_l, 1_r, 2_l, 2_r, ...]
        self.id = 1
        self.energy = '6X'
        self.mu = 0.0
        # cm
        self.sad = 1000.0
        self.iso = (0.0, 0.0, 0.0)
        self.xjaw_pos = (0.0, 0.0)
        self.yjaw_pos = (0.0, 0.0)
        # degrees
        self.gantry_rot = 0.0
        self.couch_rot = 0.0
        self.coll_rot = 0.0

    def generate(self, beamid=1):
        leaf_positions = []
        for ii in range(2):
            for edgepair in self.leaf_edges:
                leaf_positions.append(edgepair[ii]/10.0)

        s = 'Control points id (beamid.cpid) : {:d}.{:d}\n'.format(beamid, self.id) + \
            'Beam Energy Mode : {!s}\n'.format(self.energy) + \
            'MU of the current segment : {:0.6f}\n'.format(self.mu) + \
            'SAD of current beam : {:0.6f}\n'.format(self.sad/10.0) + \
            'Iso of current beam : {:0.6f}\\{:0.6f}\\{:0.6f}\n'.format(*[x/10.0 for x in self.iso]) + \
            'Gantry Rotation [Rotx, Roty, Rotz] (degrees) : {:0.6f}\\{:0.6f}\\{:0.6f}\n'.format(180.0/math.pi*0.0, 180.0/math.pi*self.gantry_rot, 180.0/math.pi*0.0) + \
            'Couch Rotation [Rotx, Roty, Rotz] (degrees) : {:0.6f}\\{:0.6f}\\{:0.6f}\n'.format(180.0/math.pi*0.0, 180.0/math.pi*0.0, 180.0/math.pi*self.couch_rot) + \
            'Collimator rotation (degrees) : {:0.6f}\n'.format(180.0/math.pi*self.coll_rot) + \
            'XJaw Position : {:0.6f}\\{:0.6f}\n'.format(*[x/10.0 for x in self.xjaw_pos]) + \
            'YJaw Position : {:0.6f}\\{:0.6f}\n'.format(*[x/10.0 for x in self.yjaw_pos]) + \
            'Number of leaf pairs : {:d}\n'.format(len(self.leaf_edges)) + \
            'Leaf positions : ' + '\\'.join(('{:0.2f}'.format(e) for e in leaf_positions)) + '\\\n\n'
        return s


class MLCBeam():
    def __init__(self):
        self.control_points = []
        self.id = 1
        self.weight_per_mu = 1e8

    @property
    def total_mu(self):
        total = 0
        for cp in self.control_points:
            total += cp.mu
        return total

from collections import deque
def get_leaf_edges(positions: List[Tuple[int, int]], fmapdims: Tuple[int, int], beamletsize: Tuple[float, float]) -> List[Tuple[float, float]]:
    """stratify binary fluence map into deliverable MLC sequences (control points)"""
    magnification = 1.0
    closed_position = tuple([-(fmapdims[ii]/2)*beamletsize[ii]-10 for ii in range(2)])

    pre = []
    post = []

    firstrow = positions[0][1]
    lastrow = positions[-1][1]
    rows = [deque() for _ in range(firstrow, lastrow+1)]

    lmark = 0
    rmark = 0
    for row in range(firstrow, lastrow+1):
        prev = positions[lmark][0]
        curr = positions[rmark][0]
        startseg = prev
        while rmark < len(positions) and positions[rmark][1] <= row:
            rmark += 1
            if rmark < len(positions):
                curr = positions[rmark][0]
                if curr-prev>1:
                    rows[row-firstrow].append(( (startseg-0.5-(fmapdims[0]/2))*beamletsize[0]*magnification,
                                                (positions[rmark-1][0]+0.5-(fmapdims[0]/2))*beamletsize[0]*magnification ))
                    startseg = curr
                prev = curr
        rows[row-firstrow].append(( (startseg-0.5-(fmapdims[0]/2))*beamletsize[0]*magnification,
                                    (positions[rmark-1][0]+0.5-(fmapdims[0]/2))*beamletsize[0]*magnification ))

        #  leaf_edges.append(((positions[lmark][0]-0.5-(fmapdims[0]/2))*beamletsize[0]*magnification,
        #                     (positions[rmark-1][0]+0.5-(fmapdims[0]/2))*beamletsize[0])*magnification)
        lmark = rmark

    if firstrow>0:
        pre += [closed_position]*(firstrow)
    post += [closed_position]*(fmapdims[1] - lastrow-1)

    segments = []
    while True:
        segmentalive = False
        leaf_edges = pre.copy()
        for row in rows:
            if len(row):
                leaf_edges.append(row.popleft())
                segmentalive = True
            else:
                leaf_edges.append(closed_position)

        if segmentalive:
            leaf_edges += post.copy()
            segments.append(leaf_edges)
        else:
            break
    return segments

def get_jaw_positions(leaf_edges: List[Tuple[float, float]], fmapdims: Tuple[int, int], beamletsize: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """set jaw position to smallest possible field without disrupting MLC opening"""
    # set xjaw
    xhalfwidth = fmapdims[0]*beamletsize[0]/2
    xjaw = [float('inf'), -float('inf')]
    for neg, pos in leaf_edges:
        if neg >= -xhalfwidth and neg < xjaw[0]:
            xjaw[0] = neg
        if pos <= xhalfwidth and pos > xjaw[1]:
            xjaw[1] = pos
    xjaw[0] = max(xjaw[0], -xhalfwidth)
    xjaw[1] = min(xjaw[1],  xhalfwidth)

    # set yjaw
    yjaw = [sign*fmapdims[1]*beamletsize[1]/2 for sign in (-1, 1)]
    for neg, pos in leaf_edges:
        if math.isclose(math.fabs(pos-neg), 0):
            yjaw[0] += beamletsize[1]
        else:
            break
    for neg, pos in reversed(leaf_edges):
        if math.isclose(math.fabs(pos-neg), 0):
            yjaw[1] -= beamletsize[1]
        else:
            break

    return tuple(xjaw), tuple(yjaw)

def generate_mlcdef(fname: str, fmapdims: Tuple[int, int], beamletsize: Tuple[float, float]) -> None:
    yedgepos = [(ii-fmapdims[1]/2)*beamletsize[1]/10.0 for ii in range(fmapdims[1]+1)]
    with open(fname, 'w') as fd:
        fd.write(
            '{:d} ! number of leaves pairs\n'.format(fmapdims[1]) +
            '{:0.1f} ! MLC leaf Plane position\n'.format(51.0) +
            '{:0.6f} ! MLC leaf thickness\n'.format(6.6) +
            '{:0.6f} ! dynamic leaf gap\n'.format(0.0) +
            '{:0.6f} ! Radius of leaf tip\n'.format(8.0) +
            '{:0.6f} ! leaf transmission\n'.format(0.015) +
            '{:0.6f},{:0.6f} ! XJawPlane location\n'.format(27.9,35.67) +
            '{:0.6f},{:0.6f} ! YJawPlane location\n'.format(36.6, 44.4) +
            '\\'.join(('{:0.2f}'.format(y) for y in yedgepos)) + '\\'
        )

def generate_rtplancps(fname: str, mlcbeams: List[MLCBeam]) -> None:
    with open(fname, 'w') as fd:
        for beam in mlcbeams:
            for cp in beam.control_points:
                fd.write(cp.generate(beam.id))

def generate_rtplan(dname: str, mlcbeams: List[MLCBeam], fsuffix=None) -> None:
    fname = pjoin(dname, 'rtplan' + ('_'+str(fsuffix) if fsuffix else '') + '.txt')
    fnamecps = pjoin(dname, 'rtplan_cps' + ('_'+str(fsuffix) if fsuffix else '') + '.txt')
    with open(fname, 'w') as fd:
        fd.write('{:d} : number of beams\n'.format(len(mlcbeams)))
        for beam in mlcbeams:
            fd.write(
                '{:s} : Beam name\n'.format('A'+str(beam.id)) +
                '{:d} : Number of Control points\n'.format(len(beam.control_points)) +
                '{:f} : Total MU\n'.format(beam.total_mu) +
                '{:e} : Weight Per MU\n'.format(beam.weight_per_mu)
            )
        fd.write('{!s} : Control Points filename'.format(os.path.basename(fnamecps)))

    generate_rtplancps(fnamecps, mlcbeams)
