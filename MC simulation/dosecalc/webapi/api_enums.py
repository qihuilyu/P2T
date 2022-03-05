# SOCKET COMM. RELATED
class MESSAGETYPE():
    """standard type that can be used in JSON payloads to steer receiver handling"""
    SIMULATE = "simulate"
    FILEPRE  = 'filepre'
    REQUEST  = 'request'  # generic request
    RESPONSE = 'response' # generic response
    STATUSREQUEST = 'statusrequest' # request a status

    DATAINGEST  = 'dataingest'
    SIMSTATUS  = "simstatus"

class STATUS():
    """Indicates a general status of server-side processing tasks"""
    SUCCESS  = 'success'
    FAILURE  = "failure"


# DATABASE RELATED
class DBCOLLECTIONS():
    IMAGES       = 'Images'
    BEAMPHOTON   = 'Beams_Photon'
    MCGEOM       = 'MC_Geometry'
    SIMULATION   = 'MC_Simulation'

class MLROLE():
    TRAIN        = 'train'
    TEST         = 'test'
    NONE         = None

class VARTYPE():
    HIGH         = 'highvar'
    LOW          = 'lowvar'

class MCGEOTYPE():
    BULKDENS     = 'bulk-density'
    INTERP       = 'interpolated'

class PARTICLETYPE():
    PHOTON       = 'photon'
    ELECTRON     = 'electron'
    PROTON       = 'proton'

class PROCSTATUS():
    FINISHED     = 'finished'
    SKIPPED      = 'skipped'
    FAILED       = 'failed'
    INPROGRESS   = 'in-progress'
    QUEUED       = 'queued'
    PENDING      = 'pending'

class STORAGETYPE():
    DENSE        = 'dense'
    SPARSE       = 'sparse'
