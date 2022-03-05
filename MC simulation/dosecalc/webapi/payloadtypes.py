import socket
from datetime import datetime
import warnings

from bson.objectid import ObjectId

import socketio
import log
from api_enums import (MCGEOTYPE, MLROLE, VARTYPE, PARTICLETYPE, STORAGETYPE)
logger = log.get_module_logger(__name__)

class BasePayload():
    def __init__(self, *args, **kwargs):
        self.payloadtype = self.__class__.__name__
        self.host = socketio.gethostname()
        self.address = None

    def __str__(self):
        return str(type(self)) + str(self.todict())

    def get(self, k, d=None):
        return self.__dict__.get(k, d)

    def send(self, addr, timeout=socket.getdefaulttimeout(), connection_timeout=socket.getdefaulttimeout()):
        """send this payload as 1-way message to the specified addr/port pair"""
        sock = socket.create_connection(addr, timeout=connection_timeout)
        sock.settimeout(timeout)
        socketio.send_payload(sock, self.todict())

    def send_request(self, addr, timeout=socket.getdefaulttimeout(), connection_timeout=socket.getdefaulttimeout()):
        """send this payload as a 2-way request to the specified addr/port pair and return the response"""
        sock = socket.create_connection(addr, timeout=connection_timeout)
        sock.settimeout(timeout)
        response = socketio.send_request(sock, self.todict(), timeout=timeout)
        return response

    def todict(self):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(k, str) and not k.startswith('_'):
                d[k] = BasePayload._convertfrompayload(v)
        return d

    @staticmethod
    def _convertfrompayload(x):
        v = x
        if isinstance(v, BasePayload):
            v = v.todict()
        elif isinstance(v, ObjectId):
            v = {'objectid': str(v)}
        elif isinstance(v, (list, tuple)):
            l = []
            for item in v:
                l.append(BasePayload._convertfrompayload(item))
            v = l
        elif isinstance(v, dict):
            d = {}
            for k, item in v.items():
                d[k] = BasePayload._convertfrompayload(item)
            v = d
        else:
            v = socketio.make_json_friendly(v)
        return v

    @staticmethod
    def _converttopayload(x):
        v = x
        if isinstance(v, dict):
            if 'payloadtype' in v:
                v = globals()[v['payloadtype']].fromdict(v)
            elif 'objectid' in v:
                v = ObjectId(v['objectid'])
            else:
                d = {}
                for k, item in v.items():
                    d[k] = BasePayload._converttopayload(item)
                v = d
        elif isinstance(v, (list, tuple)):
            l = []
            for item in v:
                l.append(BasePayload._converttopayload(item))
            v = l
        return v

    def _fromdict(self, d):
        for k, v in d.items():
            if k not in self.__dict__:
                raise KeyError('Object of type "{}" does not contain the attribute "{}"'.format(self.__class__.__name__, str(k)))
            self.__dict__[k] = BasePayload._converttopayload(v)
        return self

    @classmethod
    def fromdict(cls, d):
        if 'payloadtype' not in d:
            raise TypeError("dict does not match an existing Payload definition")
        self = globals()[d['payloadtype']]()
        return self._fromdict(d)

class TypedPayload(BasePayload):
    """intermediate type so that .fromdict() method can be used on subtypes without
    requiring a 'payloadtype' key"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def fromdict(cls, d):
        self = cls()
        return self._fromdict(d)

#==================
# Data Structures
#==================
class SimulationConfig(TypedPayload):
    """instructions for a single MC simulation execution sent with SimInstruction"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = None # ObjectId or str
        self.vartype = None # VARTYPE
        self.num_runs = 1
        self.num_particles = 1
        self.magnetic_field = None # list [x, y, z] in units: Tesla
        self.callargs = []

class SimulationResult(TypedPayload):
    """result and files from a single execution of MC code. These are collected in SimReport.simulations list
    and sent back to dataserver to be stored"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = None # ObjectId or str
        self.files = {}
        self.time_elapsed = None
        self._resultdir = None # will not be send over socket connection

class CoordSys(TypedPayload):
    def __init__(self, *args, start=None, size=None, spacing=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.start   = start
        self.size    = size
        self.spacing = spacing

    def __str__(self):
        return str({'start': self.start, 'size': self.size, 'spacing': self.spacing})

class PhotonBeam(TypedPayload):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isocenter      = None
        self.angle_gantry   = None
        self.angle_couch    = None
        self.angle_coll     = None
        self.sad            = None
        self.fmapdims       = None
        self.beamletspacing = None
        self.beamletsize    = None
        self.energy         = None
        self.beamlets       = []
        self.ptv_margin     = None
        self.particletype   = PARTICLETYPE.PHOTON
        self.gps_template   = None

#======================================
# INTERNAL COMMUNICATION PAYLOADS
#======================================
class SimInstruction(TypedPayload):
    """Instructions sent from dataserver to computeserver telling how to perform MC simulation for one or more
    simulation configs with a common subbeam (beamlet, proton spot)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_vacant_threads = 2
        self.id = None
        self.beam_id = None
        self.subbeam_id = None
        self.files = {}
        self.simulations = [] # SimulationConfig

        self.reply_host = None
        self.reply_port = None

class SimReport(TypedPayload):
    """Sent from Computeserver to dataserver with results of a set of simulations for a common subbeam_id"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status = None
        self.message = None
        self.beam_id = None
        self.subbeam_id = None
        self.simulations = [] # SimulationResult

class SummaryRequest(TypedPayload):
    """Sent from client to dataserver to get summary of the data contained in database"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class PlanStatusRequest(TypedPayload):
    """Sent from client to dataserver to get summary of treatment plan simulation status"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

#==================
# RESTFUL INTERFACE
#==================
class RESTREQTYPE():
    GET    = 'get'
    INSERT = 'insert'
    UPDATE = 'update'
    DELETE = 'delete'

class RESTDOCTYPE():
    IMAGE          = 'image'
    STRUCTURE      = 'structure'
    STRUCTUREMASK  = 'structure_mask'
    GEOMETRY       = 'geometry'
    BEAMPHOTON     = 'beamphoton'
    SIMULATION     = 'simulation'
    SAMPLE         = 'sample'

class RESTReqBase(TypedPayload):
    reqtype = None
    doctype = None

    # override send with send_request since 2-way messaging is required for REST API
    def send(self, *args, **kwargs):
        warnings.warn('Use of {}.send() over {1}.send_request() is not recommended since REST requests are expected to return a response', DeprecationWarning)
        response = self.send_request(*args, **kwargs)
        return response

class RestReqGetBase(RESTReqBase):
    reqtype = RESTREQTYPE.GET
class RestReqInsertBase(RESTReqBase):
    reqtype = RESTREQTYPE.INSERT
class RestReqUpdateBase(RESTReqBase):
    reqtype = RESTREQTYPE.UPDATE
class RestReqDeleteBase(RESTReqBase):
    reqtype = RESTREQTYPE.DELETE

#======================================================
class RestReqImageBase():
    doctype = RESTDOCTYPE.IMAGE
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_uid = None
        self.image_id = None
        self.image_doi = None

class RESTReqImageGet(RestReqImageBase, RestReqGetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RESTReqImageInsert(RestReqImageBase, RestReqInsertBase):
    """Response: {'id': <ObjectId> of new image doc}"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.doi = None
        self.files = {'ct': None, 'rtstruct': None}

class RESTReqImageUpdate(RestReqImageBase, RestReqUpdateBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RESTReqImageDelete(RestReqImageBase, RestReqDeleteBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # see RestReqImageBase for possible query parameters

#======================================================
class RestReqStructureBase():
    doctype = RESTDOCTYPE.STRUCTURE
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RESTReqStructureGet(RestReqStructureBase, RestReqGetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_id = None
        self.structure_id = None
        self.structure_name = None

class RESTReqStructureInsert(RestReqStructureBase, RestReqInsertBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_id = None
        self.name = None
        self.mask = None

class RESTReqStructureUpdate(RestReqStructureBase, RestReqUpdateBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RESTReqStructureDelete(RestReqStructureBase, RestReqDeleteBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_id = None
        self.structure_id = None

#======================================================
class RestReqGeometryBase():
    doctype = RESTDOCTYPE.GEOMETRY
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RESTReqGeometryGet(RestReqGeometryBase, RestReqGetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geom_id = None
        self.filter = None

class RESTReqGeometryInsert(RestReqGeometryBase, RestReqInsertBase):
    """
    Request:
        ct_id: ObjectId of ct entry against which to generate this geom
        coordsys: Using None for any of {'start', 'size', 'spacing'} will
                  simply inherit that value from the coordsys of the referenced ct,
                  otherwise, overwrite and use a subvolume
        geomtype: MCGEOTYPE definining how to convert HU->density
    Response: {'id': <ObjectId> of new geometry doc}"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_id = None
        self.geomtype = MCGEOTYPE.BULKDENS
        self.coordsys = {'start': None, 'size': None, 'spacing': None}

class RESTReqGeometryUpdate(RestReqGeometryBase, RestReqUpdateBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RESTReqGeometryDelete(RestReqGeometryBase, RestReqDeleteBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geom_id = None

#======================================================
class RestReqBeamPhotonBase():
    doctype = RESTDOCTYPE.BEAMPHOTON
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RESTReqBeamPhotonGet(RestReqBeamPhotonBase, RestReqGetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam_id = None
        self.recursive = False
        self.filter = None

class RESTReqBeamPhotonInsert(RestReqBeamPhotonBase, RestReqInsertBase):
    """
    Request:
        geom_id: ObjectId of geometry doc within which the beam is positioned
        structure_id: ObjectId of structure (in geom_id's referenced image doc) for which this beam is
                      treating
        mlrole: intended purpose for the beam in Deep learning environment
        beams: list of PhotonBeam objects which are defined against the specified structure/geometry
    Response: {'ids': [<ObjectId>, ...]}
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geom_id      = None
        self.structure_id = None
        self.mlrole       = MLROLE.TRAIN
        self.beams        = []

class RESTReqBeamPhotonUpdate(RestReqBeamPhotonBase, RestReqUpdateBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RESTReqBeamPhotonDelete(RestReqBeamPhotonBase, RestReqDeleteBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam_id = None

#======================================================
class RestReqSimulationBase():
    doctype = RESTDOCTYPE.SIMULATION
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RESTReqSimulationGet(RestReqSimulationBase, RestReqGetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim_id = None
        self.filter = None
        self.filedata = False

class RESTReqSimulationInsert(RestReqSimulationBase, RestReqInsertBase):
    """
    Request:
        beam_id: ObjectId of the beam doc to which to add simulation requests
        vartype: classification of dose by noise-level
        num_runs: number of independent samples to simulate for each program execution (batching)
        num_particles: number of particles to simulate for each independent run
        beamlets: select which beamlets will be simulated. Choices are:
            None: all "active" beamlets, determined at beam insertion by ray-tracing
            {'ids': [<ObjectId>, ...]}    -- selection by beamlet ObjectIds
            {'positions': [(3, 5), ...]}  -- selection of beamlets by their position in beam
            {'random-count': <int>}       -- random selection of the specified number of "active" beamlets
        magnetic_field: specify magnetic field vector as (x, y, z, 'tesla')
    Response: {'ids': [<ObjectId>, ...]}
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam_id = None
        self.vartype = VARTYPE.HIGH
        self.num_runs = 1
        self.num_particles = None
        self.beamlets = None
        self.magnetic_field = (0,0,0,'tesla')
        self.storage_type = STORAGETYPE.SPARSE
        self.sparse_threshold = 0.0
        self.callargs = []
        self.priority = None
        self.tag = None

class RESTReqSimulationUpdate(RestReqSimulationBase, RestReqUpdateBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RESTReqSimulationDelete(RestReqSimulationBase, RestReqDeleteBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter = None

#======================================================
class RestReqSampleBase():
    doctype = RESTDOCTYPE.SAMPLE
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RESTReqSampleGet(RestReqSampleBase, RestReqGetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim_id = None
        self.sample_id = None
        self.senddata = False

class RESTReqSampleInsert(RestReqSampleBase, RestReqInsertBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RESTReqSampleUpdate(RestReqSampleBase, RestReqUpdateBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RESTReqSampleDelete(RestReqSampleBase, RestReqDeleteBase):
    """
    Request:
        sim_id: ObjectId for simulation containing sample
        sample_id: ObjectId for sample to delete in referenced simulation
        decrement_nruns: update simulation to produce fewer runs in future exec.
    Response: {'sim_id': <ObjectId>, 'sample_id': <ObjectId>}
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim_id = None
        self.sample_id = None
        self.decrement_nruns = True
