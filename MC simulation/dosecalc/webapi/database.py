import sys,os
from os.path import join as pjoin
from collections import defaultdict
import shutil
from datetime import datetime
import re
import math
import struct
import json
import tempfile
import copy

import tqdm
import numpy as np
import pymongo
from pymongo import MongoClient, ReturnDocument
from bson.objectid import ObjectId
from rttypes.frame import FrameOfReference
from rttypes.volume import Volume
from rttypes.morph import binary_expansion

import socketio
import payloadtypes
from api_enums import (MESSAGETYPE, STATUS, MLROLE, PROCSTATUS,
                       VARTYPE, MCGEOTYPE, PARTICLETYPE, DBCOLLECTIONS,
                       STORAGETYPE)
from ct2mat import convert_ct_to_density
import generate_input
import dicomutils
from utils import get_directory_size
import geometry
import log

logger = log.get_module_logger(__name__)

dbclient = None
db = None
db_connect_settings = {}
def init_dbclient(host='127.0.0.1', port=27017, dbname='data', auth=None):
    """initialize once and keep open"""
    global dbclient, db, db_connect_settings
    if dbclient is None:
        if auth and len(auth)>=2:
            dbclient = MongoClient(host, port, username=auth[0], password=auth[1])
        else:
            dbclient = MongoClient(host, port)
        try:
            dbclient.server_info()
        except pymongo.errors.ServerSelectionTimeoutError as e:
            dbclient = None
            raise
        db = dbclient[dbname]
        db_connect_settings = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'auth': auth,
        }
    return dbclient, db

def reinit_dbclient():
    global dbclient, db, db_connect_settings
    for key in ('host', 'port', 'dbname', 'auth'):
        assert key in db_connect_settings
    dbclient = None
    return init_dbclient(**db_connect_settings)


DATASTORE = None
class InitDataStorage():
    def __init__(self, root='./data'):
        global DATASTORE
        self.DATAROOT  = root
        self.IMAGEDATA = pjoin(root, 'images')
        self.SIMDATA   = pjoin(root, 'simulation')
        self.MCGEODATA = pjoin(root, 'mcgeometry')
        DATASTORE = self

#====================================
# ENTRYPOINTS (external)
#====================================
def image_insert(ctfiles, rtstructfile, doi):
    # get ct metadata
    with tempfile.TemporaryDirectory() as tmpdir:
        dicomfile = socketio.unpack_file(tmpdir, ctfiles[0])
        ct_seriesid = dicomutils.get_dicom_seriesuid(dicomfile)

    image_objectid = ObjectId()
    imagedir = pjoin(DATASTORE.IMAGEDATA, str(image_objectid))

    if doi_exists(doi):
        raise FileExistsError('Image document already exists for requested doi "{}"'.format(doi))
    #  if ctrecord_exists(ct_seriesid):
        #  raise FileExistsError('CT data already exists for id:{}'.format(ct_seriesid))

    logger.debug('Unpacking CT files')
    ctdir = pjoin(imagedir, 'ct')
    ctfilenames = sorted(socketio.unpack_files(ctdir, ctfiles),
                         key=lambda x: int(re.search('.(\d+).dcm$', x).group(1)))
    if len(rtstructfile):
        rtstructfilename = socketio.unpack_file(pjoin(imagedir, 'rtstruct'), rtstructfile[0])
    else:
        rtstructfilename = None

    # load ct metadata
    frame = FrameOfReference.fromDir(ctdir)

    # pack up ct doc
    coordsys = gen_doc_coordsys(
        start=[float(x) for x in frame.start],
        size=[int(x) for x in frame.size],
        spacing=[float(x) for x in frame.spacing],
    )
    imagedoc = gen_doc_image(ctid=ct_seriesid,
                                   doi=doi,
                                   ctfiles=ctfilenames,
                                   rtstructfile=rtstructfilename,
                                   coordsys=coordsys,
                                   objectid=image_objectid)
    db[DBCOLLECTIONS.IMAGES].insert_one(imagedoc)
    return image_objectid

def image_delete(image_id):
    imagedoc = get_doc(DBCOLLECTIONS.IMAGES, image_id)
    if imagedoc is None:
        raise RuntimeError('Image "{}" couldn\'t be found'.format(image_id))

    # delete all dependent documents
    for geomdoc in db[DBCOLLECTIONS.MCGEOM].find({'image_id': ObjectId(image_id)}):
        geometry_delete(geomdoc['_id'])

    # delete image doc and any referenced files
    try:
        shutil.rmtree(build_datapath_image(image_id))
    except FileNotFoundError as err:
        logger.warning("Error while deleting image data: "+str(err))

    # remove db entry
    res = db[DBCOLLECTIONS.IMAGES].delete_many({'_id': ObjectId(image_id)})
    if res.deleted_count <= 0:
        raise RuntimeError('Failed to delete image "{}" (doi: "{}")'.format(image_id, imagedoc['doi']))
    logger.info('Deleted image matching id "{}" (doi: "{}")'.format(image_id, imagedoc['doi']))

def structure_insert(image_id, name, mask=None, bbox_buffer_mm=10):
    imagedoc = get_doc(DBCOLLECTIONS.IMAGES, image_id)
    if imagedoc is None:
        raise ValueError('Document for image "{}" couldn\'t be located'.format(image_id))
    if name in (d['name'] for d in imagedoc['structures']):
        raise ValueError('Structure "{}" already exists for image "{}"'.format(name, image_id))

    # load coordinate system from referenced imagedoc
    frame = FrameOfReference()
    frame.start = imagedoc['coordsys']['start']
    frame.size = imagedoc['coordsys']['size']
    frame.spacing = imagedoc['coordsys']['spacing']
    rtstructfile = dbabspath(imagedoc['rtstruct'])
    imagedir = os.path.dirname(os.path.dirname(dbabspath(imagedoc['images'][0])))

    # create structure doc
    maskfile = pjoin(imagedir, 'structmasks', name+'.npy')
    os.makedirs(os.path.dirname(maskfile), exist_ok=True)
    if mask is not None:
        if tuple(mask.shape[::-1]) != tuple(frame.size):
            raise RuntimeError('Array size for "{}" structure\'s provided mask array must match image size "{}", but is instead "{}"'.format(name, frame.size, mask.shape[::-1]))

        bbox = find_boundbox_from_mask(mask, frame, buffer=bbox_buffer_mm)
    else:
        mask = dicomutils.generate_mask(rtstructfile, frame, name)

        # calculate bounding box
        bbox = dicomutils.get_roi_bbox(rtstructfile, frame, name, buffer=bbox_buffer_mm)

    np.save(maskfile, mask)
    bboxdoc = gen_doc_coordsys(
        start=bbox.start,
        size=bbox.size,
        spacing=bbox.spacing,
    )

    structure_objectid = ObjectId()
    structuredoc = gen_doc_structure(
        name=name,
        centroid=dicomutils.centroid_as_coords(mask, frame),
        boundbox=bboxdoc,
        maskfile=maskfile,
        objectid=structure_objectid,
    )

    result = db[DBCOLLECTIONS.IMAGES].update_one(
        filter={'_id': ObjectId(image_id)},
        update={'$push': {'structures': structuredoc}}
    )
    if result.modified_count != 1:
        raise RuntimeError('Failed to insert new structure into image')
    return structure_objectid

def structure_delete(image_id, structure_id):
    db[DBCOLLECTIONS.IMAGES].update_one(
        filter={'_id': ObjectId(image_id)},
        update={
            '$pull': {'structures': {"_id": ObjectId(structure_id)}}
        })
    logger.info("removing structure \"{}\" from image \"{}\"".format(
        structure_id, image_id
    ))

def geometry_insert(image_id, start, size, spacing, geomtype=MCGEOTYPE.BULKDENS):
    geom_objectid = ObjectId()
    coordsys = gen_doc_coordsys(start=start,
                                size=size,
                                spacing=spacing)
    geomdoc = gen_doc_mcgeometry(image_id=image_id,
                                 geomtype=geomtype,
                                 coordsys=coordsys,
                                 objectid=geom_objectid)
    db[DBCOLLECTIONS.MCGEOM].insert_one(geomdoc)
    increment_refcount(DBCOLLECTIONS.IMAGES, image_id)
    logger.info('Added geometry "{}" to image "{}"'.format(geom_objectid, image_id))
    return geom_objectid

def geometry_delete(geom_id):
    """Delete MC Geometry and all dependent beams (recursively)"""
    geomdoc = get_doc(DBCOLLECTIONS.MCGEOM, geom_id)
    if geomdoc is None:
        raise RuntimeError('Geometry "{}" couldn\'t be found'.format(geom_id))

    # delete all beams referencing this geometry
    beamdocs = db[DBCOLLECTIONS.BEAMPHOTON].find({'geom_id': ObjectId(geom_id)})
    for beamdoc in beamdocs:
        beam_delete(beamdoc['_id'])

    # delete geometry files/folder
    try:
        shutil.rmtree(build_datapath_geom(geom_id))
    except FileNotFoundError as err:
        logger.warning('Error while deleting geometry data: '+str(err))

    # remove db entry
    res = db[DBCOLLECTIONS.MCGEOM].delete_many({'_id': ObjectId(geom_id)})
    if res.deleted_count <= 0:
        raise RuntimeError('Failed to delete geometry "{}"'.format(geom_id))
    logger.info('Deleted geometry matching id "{}"'.format(geom_id))

def beam_insert(geom_id, structure_id, angle_gantry, angle_couch, angle_coll, sad, fmapdims, beamletspacing, beamletsize, particletype, energy, isocenter=None, beamlets=None, ptv_margin=None, mlrole=MLROLE.TRAIN, gps_template=None):
    """Insert a single beam into database, referencing a geom and structure pair, and return beam_id

    Args:
        geom_id:      mongo id referencing geometry object
        structure_id: mongo id referencing structure object (nested below the image object refd. by geom.)
        angle_gantry: beam gantry angle in radians (0 rad beam shoots in +y direction; coplanar angle)
        angle_couch:  beam couch angle in radians
        angle_coll:   collimator rotation angle in radians
        sad:          Source-to-isocenter distance in [millimeters]
        fmapdims:     Beamlet/subbeam count along each axis of beam cross-section as (x, z) tuple
        beamletspacing:  Separation of beamlet centers as (x, z) tuple in [millimeters]
        beamletsize:  Physical size of each dimension of beamlet cross-section as (x, z) tuple in [millimeters]
        particletype: string indicating the type of particle used for treatment (enum PARTICLETYPE)
        energy:       energy of beam (currently for photon: ignored, for electron: monoenergy in MeV)
        isocenter:    treatment isocenter in dicom coordinates as (x, y, z) tuple in [millimeters]
                      or "None" to automatically match structure centroid coordinates
        beamlets:     Strategy for selecting beamlets for which to calculate dose
            = None  -  Raytrace and use any beamlet that intersects structure
            = [...] -  Manually supply a list of beamlets by their position in the fluence map as (y,x) tuples
            = {'random-count': <int>} -  First raytrace, then keep only "random-count" of them, selected at random
        ptv_margin:   optional expansion margin added to PTV structure mask during raytracing [mm]
                      if set to None, no margin expansion is applied
    """
    # confirm geometry exists
    geomdoc = get_doc(DBCOLLECTIONS.MCGEOM, geom_id)
    if geomdoc is None:
        raise ValueError('valid geometry document could not be found')
    density_file = pjoin(build_datapath_geom(str(geomdoc['_id'])), 'density.npy')
    imagedoc = get_doc(DBCOLLECTIONS.IMAGES, geomdoc['image_id'])
    if imagedoc is None:
        raise ValueError('valid image document could not be found')
    structures = {str(x['_id']): x for x in imagedoc['structures']}
    structuredoc = structures[str(structure_id)]

    beam_objectid = ObjectId()
    if isocenter is None:
        isocenter = structuredoc['centroid']
        logger.warning('Isocenter wasn\'t specified for beam "{!s}". Using PTV ({!s}) centroid ({!s}) instead.'.format(
            beam_objectid, structuredoc['name'], structuredoc['centroid']
        ))

    # insert beam doc
    if particletype in [PARTICLETYPE.PHOTON, PARTICLETYPE.ELECTRON]:
        beamdoc = gen_doc_photonbeam(image_id=imagedoc['_id'],
                                     geom_id=geomdoc['_id'],
                                     structure_id=structure_id,
                                     mlrole=mlrole,
                                     angle_gantry=angle_gantry,
                                     angle_couch=angle_couch,
                                     angle_coll=angle_coll,
                                     sad=sad,
                                     fmapdims=fmapdims,
                                     beamletspacing=beamletspacing,
                                     beamletsize=beamletsize,
                                     isocenter=isocenter,
                                     energy=energy,
                                     beamlets=[],
                                     ptv_margin=ptv_margin,
                                     particletype=particletype,
                                     gps_template=gps_template,
                                     objectid=beam_objectid,
                                     )
    elif particletype in [PARTICLETYPE.PROTON]:
        beamdoc = gen_doc_protonbeam(image_id=imagedoc['_id'],
                                     geom_id=geomdoc['_id'],
                                     structure_id=structure_id,
                                     mlrole=mlrole,
                                     angle_gantry=angle_gantry,
                                     angle_couch=angle_couch,
                                     angle_coll=angle_coll,
                                     sad=sad,
                                     fmapdims=fmapdims,
                                     beamletspacing=beamletspacing,
                                     beamletsize=beamletsize,
                                     isocenter=isocenter,
                                     energy=energy,
                                     beamlets=[],
                                     ptv_margin=ptv_margin,
                                     particletype=particletype,
                                     gps_template=gps_template,
                                     objectid=beam_objectid,
                                     )

    db[DBCOLLECTIONS.BEAMPHOTON].insert_one(beamdoc)
    increment_refcount(DBCOLLECTIONS.MCGEOM, geomdoc['_id'])
    logger.info('Added beam "{}" to geometry "{}"'.format(beam_objectid, geomdoc['_id']))

    # determine which beamlets to include
    beam_path = build_datapath_beam(beam_objectid)
    os.makedirs(beam_path, exist_ok=True)
    if not (isinstance(beamlets, (list, tuple)) and len(beamlets)>0):
        # raytrace
        mask = get_structure_mask(imagedoc['_id'], structuredoc['_id'], margin=ptv_margin)

        assert np.sum(mask) != 0.0
        if particletype in [PARTICLETYPE.PHOTON, PARTICLETYPE.ELECTRON]:
            active_beamlets, _ = geometry.get_active_beamlets(mask=mask,
                                                              angle_gantry=angle_gantry,
                                                              angle_couch=angle_couch,
                                                              angle_coll=angle_coll,
                                                              iso=isocenter,
                                                              start=imagedoc['coordsys']['start'],
                                                              spacing=imagedoc['coordsys']['spacing'],
                                                              fmapdims=fmapdims,
                                                              beamletspacing=beamletspacing,
                                                              beamletsize=beamletsize,
                                                              sad=sad,
                                                              vispath=beam_path,
                                                              )
        elif particletype in [PARTICLETYPE.PROTON]:
            # output should be [(position, energy), ((x, y), energy)]
            ctvol, _ = get_ctvolume(imagedoc=imagedoc)
            if os.path.isfile(density_file):
                density = np.load(density_file)
            else:
                density = convert_ct_to_density(ctvol)
                os.makedirs(os.path.dirname(density_file), exist_ok=True)
                np.save(density_file, density)
            active_beamlets = geometry.get_active_spots(density=density,
                                                        mask=mask,
                                                        angle_gantry=angle_gantry,
                                                        angle_couch=angle_couch,
                                                        angle_coll=angle_coll,
                                                        iso=isocenter,
                                                        start=imagedoc['coordsys']['start'],
                                                        spacing=imagedoc['coordsys']['spacing'],
                                                        fmapdims=fmapdims,
                                                        beamletspacing=beamletspacing,
                                                        beamletsize=beamletsize,
                                                        sad=sad,
                                                        )

        # use subset of active beamlets if requested
        if (isinstance(beamlets, dict) and 'random-count' in beamlets):
            # random subset of "active" beamlets
            active_beamlets = [active_beamlets[ii] for ii in
                        np.random.choice(len(active_beamlets),
                                         size=beamlets['random-count'],
                                         replace=False)]
    else:
        active_beamlets = beamlets

    if particletype in [PARTICLETYPE.PHOTON, PARTICLETYPE.ELECTRON]:
        subbeam_insert(beam_objectid, active_beamlets)
    elif particletype in [PARTICLETYPE.PROTON]:
        subbeam_insert_spot(beam_objectid, active_beamlets)

    return beam_objectid

def beam_delete(beam_id):
    """Delete beam and all dependent subbeams/simulations (recursively)"""
    beamdoc = get_doc(DBCOLLECTIONS.BEAMPHOTON, beam_id)
    if beamdoc is None:
        raise RuntimeError('Beam "{}" couldn\'t be found'.format(beam_id))

    # delete all subbeams
    for subbeam in beamdoc['beamlets']:
        for sim_id in subbeam['simulations']:
            try:
                simulation_delete(sim_id, remove_refs=False)
            except:
                logger.exception('Failed to delete simulation "{}"'.format(sim_id))
        try:
            shutil.rmtree(build_datapath_subbeam(beam_id, subbeam["_id"]))
        except FileNotFoundError as err:
            logger.warning('Error while deleting sub-beam data: '+str(err))

    # delete beam folder
    try:
        shutil.rmtree(build_datapath_beam(beam_id))
    except FileNotFoundError as err:
        logger.warning('Error while deleting beam data: '+str(err))

    # remove db entry
    res = db[DBCOLLECTIONS.BEAMPHOTON].delete_many({'_id': ObjectId(beam_id)})
    if res.deleted_count <=0:
        raise RuntimeError('Failed to delete beam "{}"'.format(beam_id))
    logger.info('Deleted beam matching id "{}"'.format(beam_id))

def subbeam_insert(beam_id, positions, energy=None):
    """Generate new beamlet object and append it to existing beam
    Each position should be a tuple: (pos_y, pos_x)
    """
    if not isinstance(positions, (tuple, list)):
        positions = list(positions)

    beamdoc = get_doc(DBCOLLECTIONS.BEAMPHOTON, beam_id)
    beamletdocs = []
    for position_yx in positions:
        beamlet_objectid = ObjectId()
        gpsfile = pjoin(DATASTORE.SIMDATA, str(beamdoc['_id']), str(beamlet_objectid), 'gps.mac')
        generate_gps_file(beamdoc, position_yx, gpsfile)
        beamletdoc = gen_doc_beamlet(position=position_yx,
                                     gpsfile=gpsfile,
                                     objectid=beamlet_objectid)
        beamletdocs.append(beamletdoc)

    #insert into db
    db[DBCOLLECTIONS.BEAMPHOTON].update_one(filter={'_id': ObjectId(beam_id)},
                                            update={
                                                '$push': {"beamlets": {"$each": beamletdocs, "$sort": {"position": 1}}}
                                            })
    for beamletdoc in beamletdocs:
        logger.info('Added sub-beam "{}" to beam "{}"'.format(beamletdoc['_id'], beam_id))
    return [str(doc['_id']) for doc in beamletdocs]

def subbeam_insert_spot(beam_id, positions):
    """Generate new spot object and append it to existing beam
    Each position should be a tuple: ((pos_y, pos_x), energy_MV)
    """
    if not isinstance(positions, (tuple, list)):
        positions = list(positions)

    beamdoc = get_doc(DBCOLLECTIONS.BEAMPHOTON, beam_id)
    beamletdocs = []
    for position_yx, energy in positions:
        beamlet_objectid = ObjectId()
        gpsfile = pjoin(DATASTORE.SIMDATA, str(beamdoc['_id']), str(beamlet_objectid), 'gps.mac')
        generate_gps_file(beamdoc, position_yx, gpsfile, energy=energy)
        beamletdoc = gen_doc_spot(position=position_yx,
                                  energy=energy,
                                  gpsfile=gpsfile,
                                  objectid=beamlet_objectid)
        beamletdocs.append(beamletdoc)

    #insert into db
    db[DBCOLLECTIONS.BEAMPHOTON].update_one(filter={'_id': ObjectId(beam_id)},
                                            update={
                                                '$push': {"beamlets": {"$each": beamletdocs, "$sort": {"position": 1}}}
                                            })
    for beamletdoc in beamletdocs:
        logger.info('Added sub-beam "{}" to beam "{}"'.format(beamletdoc['_id'], beam_id))
    return [str(doc['_id']) for doc in beamletdocs]

def subbeam_delete(beam_id, subbeam_id, beamdoc=None):
    """Delete a bemalet from the specified beam. Also deletes all simulations associated with the deleted
    beamlet"""
    if beamdoc is None:
        beamdoc = get_doc(DBCOLLECTIONS.BEAMPHOTON, beam_id)
    beamletdoc = None
    for doc in beamdoc['beamlets']:
        if ObjectId(doc['_id']) == ObjectId(subbeam_id):
            beamletdoc = doc
            break
    if beamletdoc is None:
        raise RuntimeError("Beamlet \"{}\" couldn't be located".format(subbeam_id))

    for sim_id in beamletdoc['simulations']:
        try:
            simulation_delete(sim_id, remove_refs=False)
        except:
            logger.exception('Failed to delete simulation "{}"'.format(sim_id))
    try:
        shutil.rmtree(build_datapath_subbeam(beam_id, subbeam_id))
    except FileNotFoundError as err:
        logger.warning('Error while deleting sub-beam data: '+str(err))
    res = db[DBCOLLECTIONS.BEAMPHOTON].update_many({'_id': ObjectId(beam_id)},
                                            update={
                                                "$pull": {'beamlets': {'_id': ObjectId(subbeam_id)}}
                                            })
    if res.modified_count<=0:
        raise RuntimeError('Failed to delete sub-beam "{}"'.format(subbeam_id))
    logger.info('Deleted {} sub-beam "{}" from beam "{}"'.format(res.modified_count, subbeam_id, beam_id))

def simulation_insert(beam_id, subbeam_id, **kwargs):
    """insert a simulation document and link to the specified subbeam document"""
    beamdoc = get_doc(DBCOLLECTIONS.BEAMPHOTON, beam_id)

    sim_id = ObjectId()
    simdoc = gen_doc_simulation(image_id=beamdoc['image_id'],
                                geom_id=beamdoc['geom_id'],
                                beam_id=beam_id,
                                subbeam_id=subbeam_id,
                                objectid=sim_id,
                                **kwargs)
    # insert ref_id into beamlet doc
    result = db[DBCOLLECTIONS.BEAMPHOTON].update_one(
        filter={'_id': ObjectId(beam_id)},
        update={'$push': {'beamlets.$[beamlet].simulations': ObjectId(sim_id)}},
        array_filters=[{'beamlet._id': ObjectId(subbeam_id)}, ]
    )
    if result.modified_count != 1:
        raise RuntimeError('Problem adding simulation to sub-beam "{}"'.format(str(subbeam_id)))
    result = db[DBCOLLECTIONS.SIMULATION].insert_one(simdoc)
    logger.info('Inserted simulation "{}" for sub-beam "{}" in beam "{}"'.format(sim_id, subbeam_id, beam_id))
    return sim_id

def simulation_insert_bundled(beam_id, subbeam_ids, **kwargs):
    """Bundled version of simulation_insert() which adds sims to multiple beamlets belonging to a single beam
    with a single database call. This is much faster than iterating through beamlets and calling
    simulation_insert() in the loop

    **kwargs: see database.gen_doc_simulation()"""
    beamdoc = db[DBCOLLECTIONS.BEAMPHOTON].find_one({'_id': ObjectId(beam_id)})

    subbeam_map = {}
    for subbeam in beamdoc['beamlets']:
        subbeam_map[str(subbeam['_id'])] = subbeam

    # modify local beamdoc, then replace db doc with updated local doc outside loop
    simdocs = []
    for subbeam_id in subbeam_ids:
        subbeam_doc = subbeam_map[str(subbeam_id)]

        sim_id = ObjectId()
        simdoc = gen_doc_simulation(image_id=beamdoc['image_id'],
                                    geom_id=beamdoc['geom_id'],
                                    beam_id=beam_id,
                                    subbeam_id=subbeam_id,
                                    objectid=sim_id,
                                    **kwargs)
        simdocs.append(simdoc)
        subbeam_map[str(subbeam_id)]['simulations'].append(simdoc['_id'])

    # add simulation docs to database in one transaction
    result = db[DBCOLLECTIONS.SIMULATION].insert_many(simdocs, ordered=False)
    if len(result.inserted_ids) != len(simdocs):
        raise RuntimeError('Problem while adding {} simulations to beam "{}"'.format(len(simdocs), beam_id))
    logger.info('Inserted simulations for {} sub-beams in beam "{}"'.format(len(simdocs), beam_id))

    # update beam document with simdoc ids
    beamdoc['beamlets'] = sorted(subbeam_map.values(), key=lambda doc: doc['position'][0]*beamdoc['fmapdims'][0]+doc['position'][1])
    result = db[DBCOLLECTIONS.BEAMPHOTON].replace_one({'_id': ObjectId(beam_id)}, beamdoc)
    if result.modified_count <= 0:
        raise RuntimeError('Problem while updating beam "{}" with references to new simulations'.format(beam_id))

    return [str(doc['_id']) for doc in simdocs]

def simulation_delete(sim_id, remove_refs=True):
    """Delete simulation object from database and any accompanying data on disk

    If remove_refs is True (default), simdoc references will be removed from all
    containing subbeams docs. This is time consuming so remove_refs should be set
    to False when all containing subbeams are guaranteed to be deleted during the
    same atomic database operation.
    """
    sim_id = ObjectId(sim_id)

    # delete files from disk
    treepath = build_datapath_simulation(sim_id)
    try:
        shutil.rmtree(treepath)
    except FileNotFoundError as err:
        logger.debug('simulation folder not deleted because it doesn\'t exist: '+str(err))

    if remove_refs:
        try:
            simdoc = db[DBCOLLECTIONS.SIMULATION].find_one({'_id': sim_id})
            if not simdoc:
                raise ValueError('sim_id {!s} doesn\'t match any sim docs in db'.format(sim_id))

            beam_filter = {'_id': ObjectId(simdoc['beam_id'])}
            beamdoc = db[DBCOLLECTIONS.BEAMPHOTON].find_one(beam_filter)
            if not beamdoc:
                raise ValueError('beam_id {!s} doesn\'t match any beam docs in db'.format(simdoc['beam_id']))

            changed = False
            for ii, subbeamdoc in enumerate(beamdoc['beamlets']):
                if ObjectId(subbeamdoc['_id']) == ObjectId(simdoc['subbeam_id']) \
                        and sim_id in subbeamdoc['simulations']:
                    subbeamdoc['simulations'].remove(sim_id)
                    beamdoc['beamlets'][ii] = subbeamdoc
                    changed = True
                    logger.debug('removed sim ref {!s} from beam {!s} subbeam {!s}'.format(
                        sim_id, beamdoc['_id'], subbeamdoc['_id']))

            if changed:
                result = db[DBCOLLECTIONS.BEAMPHOTON].replace_one(
                    filter=beam_filter,
                    replacement=beamdoc)
                if not result.modified_count:
                    raise RuntimeError("Failed to replace beam doc {!s}}".format(beamdoc['_id']))
        except Exception as err:
            logger.warning("Error while removing sim doc references from database:\n{!s}".format(err))

    # remove db entry
    res = db[DBCOLLECTIONS.SIMULATION].delete_many({'_id': sim_id})
    if res.deleted_count<=0:
        raise RuntimeError('Failed to delete simulation "{}"'.format(sim_id))
    logger.info('Deleted {} simulation matching id: "{}"'.format(res.deleted_count, sim_id))

def build_datapath_image(image_id):
    return pjoin(dbabspath(DATASTORE.IMAGEDATA), str(image_id))

def build_datapath_geom(geom_id):
    return pjoin(dbabspath(DATASTORE.MCGEODATA), str(geom_id))

def build_datapath_beam(beam_id):
    return pjoin(dbabspath(DATASTORE.SIMDATA), str(beam_id))

def build_datapath_subbeam(beam_id, subbeam_id):
    return pjoin(build_datapath_beam(beam_id), str(subbeam_id))

def build_datapath_simulation(sim_id):
    simdoc = get_doc(DBCOLLECTIONS.SIMULATION, sim_id)
    return pjoin(build_datapath_subbeam(simdoc['beam_id'], simdoc['subbeam_id']), str(simdoc['_id']) )

def build_datapath_sample(sim_id, sample_id):
    return pjoin(build_datapath_simulation(sim_id), str(sample_id))

def register_generated_geometry(geom_id, geomfile):
    """match to 'empty' geometry doc in db, generate geomtery file, and link to doc"""
    db[DBCOLLECTIONS.MCGEOM].update_many(
        {'_id': ObjectId(geom_id)},
        {'$set': {'geomfile': dbrelpath(geomfile)}}
    )

def add_sims_to_beam(beam_id, **kwargs):
    """create new simulation doc and attach to all beamlets under a single beam"""
    beamcollection = db[DBCOLLECTIONS.BEAMPHOTON]
    filter = {'_id': ObjectId(beam_id)}
    beamdoc = beamcollection.find_one(filter)
    simulation_insert_bundled(beam_id=beamdoc['_id'],
                              subbeam_ids=[subbeam['_id'] for subbeam in beamdoc['beamlets']],
                              )
    for subbeam in beamdoc['beamlets']:
        simulation_insert(beam_id=beamdoc['_id'],
                          subbeam_id=subbeam['_id'],
                          **kwargs,
                          )

def generate_simulation_payload(sim_id):
    """package a single simluation or a set of simulations for a common beamlet to send to computeserver"""
    simdoc = db[DBCOLLECTIONS.SIMULATION].find_one({'_id': ObjectId(sim_id)})
    if simdoc is None:
        raise RuntimeError('simulation "{}" no longer exists.'.format(sim_id))
    simdocs = list(db[DBCOLLECTIONS.BEAMPHOTON].aggregate([
        {'$match': {'_id': simdoc['beam_id']}},
        {'$unwind': '$beamlets'},
        {'$match': {'beamlets._id': simdoc['subbeam_id']}},
        {'$lookup': {
            'from': DBCOLLECTIONS.MCGEOM,
            'localField': 'geom_id',
            'foreignField': '_id',
            'as': 'geom' }},
        {'$addFields': {'sim': [simdoc]}},
    ]))

    geometry_file = dbabspath(simdocs[0]['geom'][0]['geomfile'])
    gps_file = dbabspath(simdocs[0]['beamlets']['gpsfile'])
    beamlet_id = str(simdocs[0]['beamlets']['_id'])

    simconfigs = []
    for simdoc in simdocs:
        _simdoc = simdoc['sim'][0]
        if _simdoc['num_runs'] < 1:
            logger.warning('simulation "{!s}" ignored, # runs requested is less than 1'.format(_simdoc['_id']))
            continue
        simconfig = payloadtypes.SimulationConfig()
        simconfig.id = str(_simdoc['_id'])
        simconfig.num_particles = _simdoc['num_particles']
        simconfig.num_runs = _simdoc['num_runs'] # TODO: set to difference between requested and existing
        simconfig.vartype = _simdoc['vartype']
        simconfig.magnetic_field = _simdoc['magnetic_field']
        simconfig.callargs += _simdoc['callargs']
        if _simdoc['storage_type'] == STORAGETYPE.SPARSE:
            simconfig.callargs += ['--sparse', "--sparse-threshold", str(_simdoc['sparse_threshold'])]

        simconfigs.append(simconfig)

    if not simconfigs:
        raise RuntimeError("No simulation configurations could be formed.")

    payload = payloadtypes.SimInstruction()
    payload.id = beamlet_id
    payload.num_vacant_threads = 2
    payload.beam_id = str(simdocs[0]['_id'])
    payload.subbeam_id = beamlet_id
    payload.files = {
        'geometry': socketio.pack_file_text(name='mcgeo.txt', file=geometry_file),
        'gps': socketio.pack_file_text(name='gps.mac', file=gps_file),
    }
    payload.simulations = simconfigs
    return payload

def process_skipped_simulation(sim_id):
    """mark a simulation task as skipped, without sending to computeserver"""
    logger.debug('simulation task "{}" was skipped (num_particles <= 0)'.format(sim_id))
    simdoc = get_doc(DBCOLLECTIONS.SIMULATION, sim_id)
    if simdoc is None:
        raise RuntimeError('simulation "{}" no longer exists'.format(sim_id))

    update_doc_status(DBCOLLECTIONS.SIMULATION, sim_id, PROCSTATUS.SKIPPED, "skipped due to \"num_particles <= 0\"", 0)

def process_completed_simulation(sim, beam_id, subbeam_id, outputdir, status, message):
    """take a SIMSTATUS payload and update simulation calc. statuses, generate samples and attach to simdocs in db"""
    assert(isinstance(sim, payloadtypes.SimulationResult))
    sim_id = sim.id
    logger.debug('Processing completed simulation for "{}"'.format(sim_id))
    simdoc = get_doc(DBCOLLECTIONS.SIMULATION, sim_id)
    if simdoc is None:
        raise RuntimeError('simulation "{}" no longer exists'.format(sim_id))

    # update sim status
    procstatus = PROCSTATUS.FINISHED if status==STATUS.SUCCESS else PROCSTATUS.FAILED
    update_doc_status(DBCOLLECTIONS.SIMULATION, sim_id, procstatus, message, sim.time_elapsed)

    # unpack simulation-level files
    # only keep one density file per MCGeometry. If none exists yet, add this file to database, otherwise just
    # lookup path and store in this sample (should save considerable storage space)
    # sim-level files are in sim.files[None]
    densfile = None
    ignored_files = ['info.txt', 'run_log.txt']
    for fspec in list(sim.files.get(None, [])):
        if 'density' in fspec['name'].lower():
            densfile = pjoin(build_datapath_geom(simdoc['geom_id']), fspec['name'])
            if not os.path.isfile(densfile):
                os.makedirs(os.path.dirname(densfile), exist_ok=True)
                with open(densfile, 'wb') as fd:
                    fd.write(socketio.unpack_file_to_memory(fspec))
            sim.files[None].remove(fspec)

        elif fspec['name'] in ignored_files:
            sim.files[None].remove(fspec)

    # unpack remaining files (if any)
    socketio.unpack_files(outputdir, sim.files.get(None, []))
    if None in sim.files:
        del sim.files[None]

    # unpack per-run/sample files
    #   remaining entries in sim.files are collections of files from each sim run
    #   label is the name of the run output folder (e.g. run001)
    sampledocs = []
    for label, files in sim.files.items():
        sample_id = ObjectId()
        runoutdir = pjoin(outputdir, str(sample_id))
        fnames = socketio.unpack_files(runoutdir, files)
        dosefile = next(x for x in fnames if 'dose' in x.lower())
        try: fluencefile = next(x for x in fnames if 'fluence' in x.lower())
        except: fluencefile = None
        sampledocs.append(gen_doc_sample(dosefile=dosefile, densfile=densfile, fluencefile=fluencefile, objectid=sample_id))
    db[DBCOLLECTIONS.SIMULATION].find_one_and_update(
        {'_id': ObjectId(sim_id)},
        {'$set': {'samples': sampledocs}}
    )

def reset_simulation_record(sim_id):
    """reset simulation to a state before any computation (setup to recompute it)"""
    update_doc_status(DBCOLLECTIONS.SIMULATION, sim_id, PROCSTATUS.PENDING)
    simdoc = db[DBCOLLECTIONS.SIMULATION].find_one_and_update(
        filter={'_id': ObjectId(sim_id)},
        update={'$set': {'samples': []}},
    )
    if simdoc is None:
        raise RuntimeError('Couldn\'t update simulation "{}"'.format(sim_id))

    # remove existing files
    simdir = dbabspath(pjoin('simulation', str(simdoc['beam_id']), str(simdoc['subbeam_id']), str(simdoc['_id'])))
    #  logger.debug('removing contents of directory "{}"'.format(simdir))
    try:
        shutil.rmtree(simdir, ignore_errors=True)
    except Exception as e:
        logger.exception('failed to clean directory: "{}"'.format(simdir))

def test_simresult_validity(sim_id, persamplesize):
    simdoc = get_doc(DBCOLLECTIONS.SIMULATION, sim_id)
    if not len(simdoc['samples']):
        #  logger.debug('failed sample existence req.')
        return False
    if simdoc['storage_type'] == STORAGETYPE.DENSE:
        for sample in simdoc['samples']:
            dirsize = get_directory_size(os.path.dirname(dbabspath(sample['dosefile'])))
            #  logger.debug('dirsize: {} | expected: {}'.format(dirsize, persamplesize))
            if dirsize < persamplesize:
                #  logger.debug('failed sample size req.')
                return False
    for fname in ['beamon.in']:
        dirname = pjoin(os.path.dirname(dbabspath(simdoc['samples'][0]['dosefile'])), os.path.pardir, fname)
        if not os.path.isfile(dirname):
            #  logger.debug('failed file existence req. "{}"'.format(dirname))
            return False
    return True

def cleandb_reset_corrupt_sims(dryrun=False):
    """Iterate simulation docs in database and check each for 'validity' (expected file existence and size)

    if 'validity' is not confirmed, reset simulation doc to its initial state for re-computation
    """
    if dryrun:
        logger.info("PERFORMING DATABASE CLEANUP DRY-RUN ONLY. No changes to filesystem or database entries will be performed")
    aggcursor = db[DBCOLLECTIONS.SIMULATION].aggregate([
        {'$match': {'procstatus.status': {'$in': [PROCSTATUS.FINISHED]}}},
        {'$group': {'_id': '$geom_id', 'simids': {'$push': '$_id'}}}
    ])
    num_checked = 0
    num_cleaned = 0
    for geomgroup in aggcursor:
        geomdoc = get_doc(DBCOLLECTIONS.MCGEOM, geomgroup['_id'])
        nvoxels = np.product(geomdoc['coordsys']['size'])
        expected_persamplesize = struct.calcsize('d')*nvoxels
        num_checked += len(geomgroup['simids'])

        num_cleaned_geom = 0
        for sim_id in geomgroup['simids']:
            if not test_simresult_validity(sim_id, expected_persamplesize):
                if dryrun:
                    logger.debug('simulation "%s" would be reset', sim_id)
                else:
                    logger.debug('simulation "%s" was reset', sim_id)
                    reset_simulation_record(sim_id)
                num_cleaned_geom += 1
        logger.info('Reset {} simulations referencing geometry "{!s}"'.format(num_cleaned_geom, geomdoc['_id']))
        num_cleaned += num_cleaned_geom
    logger.info('Reset {} of {} checked simulations'.format(num_cleaned, num_checked))

def cleandb_remove_leftover_files(dryrun=False):
    """delete files which are not referenced in database (leftovers)"""
    # delete dangling/leftover images
    delcount = 0
    for image_doc in db[DBCOLLECTIONS.IMAGES].find():
        if db[DBCOLLECTIONS.MCGEOM].find_one({'image_id': ObjectId(image_doc["_id"])}) is None:
            image_delete(image_doc['_id'])
            delcount += 1
    logger.info('deleted {} dangling image docs'.format(delcount))

    # compare at the image level
    indexed_images = set()
    logger.info("Indexing image documents")
    icount = 0
    for imagedoc in db[DBCOLLECTIONS.IMAGES].find():
        indexed_images.add(str(imagedoc['_id']))
        icount += 1

    root = dbabspath(DATASTORE.IMAGEDATA)
    dircount = 0
    delcount = 0
    for d in os.listdir(root):
        if not os.path.isdir(pjoin(root, d)): continue
        dircount += 1
        if d not in indexed_images:
            delcount += 1
            if dryrun:
                logger.debug('image data not in database index: "{}"'.format(d))
            else:
                delpath = pjoin(root, d)
                try:
                    shutil.rmtree(delpath)
                    logger.debug('deleted leftover image data at "{}"'.format(delpath))
                except (FileNotFoundError, NotADirectoryError) as err:
                    logger.warning('Error while deleting unreferenced image data from "{}"'.format(delpath))
    logger.info('Indexed {} image documents'.format(icount))
    logger.info('Deleted {} dirs of {}'.format(delcount, dircount))

    #----------------
    # delete dangling/leftover geometries
    delcount = 0
    for geomdoc in db[DBCOLLECTIONS.MCGEOM].find():
        if db[DBCOLLECTIONS.IMAGES].find_one({'_id': ObjectId(geomdoc["image_id"])}) is None:
            geometry_delete(str(geomdoc['_id']))
            delcount += 1
    logger.info('deleted {} dangling geometry docs'.format(delcount))

    # compare at the geometry level
    indexed_geoms = set()
    logger.info("Indexing geometry documents")
    icount = 0
    for geomdoc in db[DBCOLLECTIONS.MCGEOM].find():
        indexed_geoms.add(str(geomdoc['_id']))
        icount += 1

    root = dbabspath(DATASTORE.MCGEODATA)
    dircount = 0
    delcount = 0
    for d in os.listdir(root):
        if not os.path.isdir(pjoin(root, d)): continue
        dircount += 1
        if d not in indexed_geoms:
            delcount += 1
            if dryrun:
                logger.debug('geometry data not in database index: "{}"'.format(d))
            else:
                delpath = pjoin(root, d)
                try:
                    shutil.rmtree(delpath)
                    logger.debug('deleted leftover geometry data at "{}"'.format(delpath))
                except (FileNotFoundError, NotADirectoryError) as err:
                    logger.warning('Error while deleting unreferenced geometry data from "{}"'.format(delpath))
    logger.info('Indexed {} geometry documents'.format(icount))
    logger.info('Deleted {} dirs of {}'.format(delcount, dircount))

    #----------------
    # delete dangling/leftover beams
    delcount = 0
    for beam_doc in db[DBCOLLECTIONS.BEAMPHOTON].find():
        if db[DBCOLLECTIONS.MCGEOM].find_one({'_id': ObjectId(beam_doc["geom_id"])}) is None \
        or db[DBCOLLECTIONS.IMAGES].find_one({'_id': ObjectId(beam_doc['image_id'])}) is None:
            beam_delete(str(beam_doc['_id']))
            delcount += 1
    logger.info('deleted {} dangling beam docs'.format(delcount))

    # compare at the beam level
    indexed_beams = set()
    logger.info("Indexing beam documents")
    icount = 0
    for beamdoc in db[DBCOLLECTIONS.BEAMPHOTON].find():
        indexed_beams.add(str(beamdoc['_id']))
        icount+=1

    root = dbabspath(DATASTORE.SIMDATA)
    dircount = 0
    delcount = 0
    for d in os.listdir(root):
        if not os.path.isdir(pjoin(root, d)): continue
        dircount+=1
        if d not in indexed_beams:
            delcount += 1
            if dryrun:
                logger.debug('beam data not in database index: "{}"'.format(d))
            else:
                delpath = pjoin(root, d)
                try:
                    shutil.rmtree(delpath)
                    logger.debug('deleted leftover beam data at "{}"'.format(delpath))
                except (FileNotFoundError, NotADirectoryError) as err:
                    logger.warning('Error while deleting unreferenced beam data from "{}": '.format(delpath)+str(err))

    logger.info('Indexed {} beam documents'.format(icount))
    logger.info('Deleted {} dirs of {}'.format(delcount, dircount))
    assert icount == dircount - delcount

    # compare at the beamlet level
    for beamid in os.listdir(root):
        indexed_subbeams = set()
        beamdoc = db[DBCOLLECTIONS.BEAMPHOTON].find_one({'_id': ObjectId(beamid)})
        if beamdoc is None: continue #temporary until leftover beam is deleted
        for subbeamdoc in beamdoc['beamlets']:
            indexed_subbeams.add(str(subbeamdoc['_id']))

        for d in os.listdir(pjoin(root, beamid)):
            if not os.path.isdir(pjoin(root, beamid, d)): continue
            if d not in indexed_subbeams:
                if dryrun:
                    logger.debug('subbeam data not in database index for subbeam "{}:{}"'.format(beamid, d))
                else:
                    delpath = pjoin(root, beamid, d)
                    try:
                        shutil.rmtree(delpath)
                        logger.debug('deleted leftover subbeam data at "{}"'.format(delpath))
                    except (FileNotFoundError, NotADirectoryError) as err:
                        logger.warning('Error while deleting unreferenced subbeam data from "{}": '.format(delpath)+str(err))

    #----------------
    # delete dangling/leftover simulations
    delcount = 0
    for sim_doc in db[DBCOLLECTIONS.SIMULATION].find():
        if db[DBCOLLECTIONS.BEAMPHOTON].find_one({'_id': ObjectId(sim_doc["beam_id"])}) is None \
        or db[DBCOLLECTIONS.MCGEOM].find_one({'_id': ObjectId(sim_doc["geom_id"])}) is None \
        or db[DBCOLLECTIONS.IMAGES].find_one({'_id': ObjectId(sim_doc["image_id"])}) is None:
            simulation_delete(str(sim_doc['_id']), remove_refs=False)
            delcount += 1
    logger.info('deleted {} dangling simulation docs'.format(delcount))

def generate_gps_file(beamdoc, position_yx, gpsfile, energy=None):
    geomdoc = get_doc(DBCOLLECTIONS.MCGEOM, beamdoc['geom_id'])
    gcs = geomdoc['coordsys']

    particletype = beamdoc['particletype']
    if energy is None:
        energy = beamdoc['energy']

    # GENERATE GPSFILE
    os.makedirs(os.path.dirname(gpsfile), exist_ok=True)
    source, focus, _ = geometry.calculate_gps_coordinates(position=position_yx,
                                                       angle_gantry=beamdoc['angle_gantry'],
                                                       angle_couch=beamdoc['angle_couch'],
                                                       angle_coll=beamdoc['angle_coll'],
                                                       iso=beamdoc['isocenter'],
                                                       start=gcs['start'],
                                                       size=gcs['size'],
                                                       spacing=gcs['spacing'],
                                                       fmapdims=beamdoc['fmapdims'],
                                                       beamletspacing=beamdoc['beamletspacing'],
                                                       beamletsize=beamdoc['beamletsize'],
                                                       sad=beamdoc['sad'],
                                                       )
    with open(gpsfile, 'w') as fd:
        if particletype == PARTICLETYPE.PHOTON:
            generate_input.generate_gps_photon(fd, source=(*source, 'mm'), focus=(*focus, 'mm'),
                                               angle_gantry=beamdoc['angle_gantry'],
                                               angle_couch=beamdoc['angle_couch'],
                                               angle_coll=beamdoc['angle_coll'],
                                               beamletsize=(*beamdoc['beamletsize'], 'mm'),
                                               sad=beamdoc['sad'], sfd=0.1*beamdoc['sad'],
                                               energy_mev=energy,
                                               desc='[{}] beamlet'.format(','.join([str(x) for x in position_yx])),
                                               gps_template=beamdoc['gps_template'],
                                               )
        elif particletype == PARTICLETYPE.ELECTRON:
            generate_input.generate_gps_electron(fd, source=(*source, 'mm'), focus=(*focus, 'mm'),
                                                 angle_gantry=beamdoc['angle_gantry'],
                                                 angle_couch=beamdoc['angle_couch'],
                                                 angle_coll=beamdoc['angle_coll'],
                                                 beamletsize=(*beamdoc['beamletsize'], 'mm'),
                                                 sad=beamdoc['sad'], sfd=0.1*beamdoc['sad'],
                                                 energy_mev=float(energy),
                                                 desc='[{}] beamlet'.format(','.join([str(x) for x in position_yx])),
                                                 gps_template=beamdoc['gps_template'],
                                                 )
        elif particletype == PARTICLETYPE.PROTON:
            generate_input.generate_gps_proton(fd, source=(*source, 'mm'), focus=(*focus, 'mm'),
                                                 angle_gantry=beamdoc['angle_gantry'],
                                                 angle_couch=beamdoc['angle_couch'],
                                                 angle_coll=beamdoc['angle_coll'],
                                                 beamletsize=(*beamdoc['beamletsize'], 'mm'),
                                                 sad=beamdoc['sad'], sfd=0.1*beamdoc['sad'],
                                                 energy_mev=float(energy),
                                                 desc='[{}] beamlet'.format(','.join([str(x) for x in position_yx])),
                                                 gps_template=beamdoc['gps_template'],
                                                 )

        else:
            raise ValueError('particletype "{}" is not supported'.format(particletype))

def regenerate_gps_file(beam_id, beamlet_id):
    beamdoc = db[DBCOLLECTIONS.BEAMPHOTON].aggregate([
        {'$match': {'_id': ObjectId(beam_id)}},
        {'$unwind': '$beamlets'},
        {'$match': {'beamlets._id': ObjectId(beamlet_id)}}
    ]).next()
    beamletdoc = beamdoc['beamlets']
    try:
        gpsfile = dbabspath(beamletdoc['gpsfile'])
        shutil.move(gpsfile, gpsfile+'.bak')
    except: pass
    generate_gps_file(beamdoc, beamletdoc['position'], gpsfile)

def regenerate_gps_files():
    cursor = db[DBCOLLECTIONS.BEAMPHOTON].find()
    for ii, beamdoc in enumerate(cursor):
        logger.debug('{:04d}| beam "{!s}"'.format(ii, beamdoc['_id']))
        for jj, beamletdoc in enumerate(beamdoc['beamlets']):
            logger.debug('  {:05d}| beamlet "{!s}"'.format(jj, beamletdoc['_id']))
            logger.debug('  Regenerating GPS file "{}"'.format(beamletdoc['gpsfile']))
            regenerate_gps_file(beamdoc['_id'], beamletdoc['_id'])
            for kk, sim_id in enumerate(beamletdoc['simulations']):
                simdoc = get_doc(DBCOLLECTIONS.SIMULATION, sim_id)
                logger.debug('     {:05d}| sim "{!s}"'.format(kk, simdoc['_id']))
                logger.debug('     resetting sim "{!s}"'.format(simdoc['_id']))
                reset_simulation_record(sim_id)

def diff_fluence_arrays(arr1, arr2, verbose=True):
    global print
    _print = print
    def print(*args, **kwargs):
        if verbose:
            _print(*args, **kwargs)

    # compare
    ma, mb = 0, 0
    ua, ub = 0, 0
    uc = 0
    while ma<len(arr1) and mb<len(arr2):
        if ma >= len(arr1):
            print(' '*8 + '  {!s:8s}'.format(arr2[mb]))
            mb+=1
            ub+=1
            continue
        elif mb >= len(arr2):
            print('{!s:8s}'.format(arr1[ma]))
            ma+=1
            ua+=1
            continue

        a = arr1[ma]
        b = arr2[mb]
        if a == b:
            print('{!s:8s}  {!s:8s}'.format(a, b))
            ma += 1
            mb += 1
            uc += 1
            continue
        elif a<b:
            print('{!s:8s}'.format(a))
            ma += 1
            ua+=1
            continue
        elif a>b:
            print(' '*8 + '  {!s:8s}'.format(b))
            mb += 1
            ub+=1
            continue
    print('{} unique to new'.format(ua))
    print('{} unique to old'.format(ub))
    print('{} shared'.format(uc))
    return (ua==0 and ub==0)

def make_fluence_array(positions, shape):
    arr = np.zeros(shape)
    for p in positions:
        arr[tuple(p)] = 1.0
    return arr

def get_data_breakdown(image_id):
    breakdown = {}

    # beam summary
    beamdocs = list(db[DBCOLLECTIONS.BEAMPHOTON].find({'image_id': ObjectId(image_id)}))
    train_beamdocs = [x for x in beamdocs if x['mlrole']==MLROLE.TRAIN]
    test_beamdocs  = [x for x in beamdocs if x['mlrole']==MLROLE.TEST]
    breakdown['num_beams'] = {'train': len(train_beamdocs),
                              'test':  len(test_beamdocs), }

    # beamlet summary
    breakdown['num_beamlets'] = {}
    breakdown['simulations'] = {}
    for role, beamdocs in (('train', train_beamdocs), ('test', test_beamdocs)):
        num_beamlets = 0
        breakdown['simulations'][role] = {}
        procstatus_counts       = defaultdict(lambda: 0)
        nhistory_simcounts      = defaultdict(lambda: 0)
        nhistory_samplecounts   = defaultdict(lambda: 0)
        magfield_simcounts      = defaultdict(lambda: 0)

        for beamdoc in beamdocs:
            num_beamlets += len(beamdoc['beamlets'])
            sim_ids = [sim_id for beamlet in beamdoc['beamlets'] for sim_id in beamlet['simulations']]
            cmdcursor = db[DBCOLLECTIONS.SIMULATION].aggregate([
                {'$match': {'_id': {'$in': sim_ids}}},
                {'$facet': {
                    'procstatus_counts': [
                        {'$group': {
                            '_id': '$procstatus.status',
                            'count': {'$sum': 1},
                        }} ],
                    'nhistory_counts': [
                        {'$group': {
                            '_id': '$num_particles',
                            'sim_count': {'$sum': 1},
                            'sample_count': {'$sum': '$num_runs'},
                        }} ],
                    'magfield_counts': [
                        {'$group': {
                            '_id': '$magnetic_field',
                            'sim_count': {'$sum': 1},
                        }} ],
                }}
            ])
            agg = next(cmdcursor)
            for procstatus in agg['procstatus_counts']:
                procstatus_counts[procstatus['_id']] += procstatus['count']
            for nhistory in agg['nhistory_counts']:
                nhistory_simcounts[nhistory['_id']] += nhistory['sim_count']
                nhistory_samplecounts[nhistory['_id']] += nhistory['sample_count']
            for nmag in agg['magfield_counts']:
                if nmag['_id'] is not None:
                    magfield_simcounts[str(nmag['_id'][2])+'T'] += nmag['sim_count']
                else:
                    magfield_simcounts[nmag['_id']] += nmag['sim_count']
        breakdown['simulations'][role]['procstatus_counts']      = dict(procstatus_counts)
        breakdown['simulations'][role]['nhistory_sim_counts']    = dict(nhistory_simcounts)
        breakdown['simulations'][role]['nhistory_sample_counts'] = dict(nhistory_samplecounts)
        breakdown['simulations'][role]['magfield_sim_counts']    = dict(magfield_simcounts)
        breakdown['num_beamlets'][role] = num_beamlets

    return breakdown

def get_summary():
    images = db[DBCOLLECTIONS.IMAGES].find()
    summary = {}
    for image in images:
        summary[image['doi']] = get_data_breakdown(image['_id'])
    return summary

def db_field_ensure_exists(collection, field, default, filter=None):
    dbfilter = {field: {"$exists": False}}
    if filter is not None:
        assert isinstance(filter, dict)
        dbfilter = {**dbfilter, **filter}
    updated_count = db[collection].update_many(
        filter=dbfilter,
        update={'$set': {field: default}}
    ).matched_count
    if updated_count > 0:
        logger.debug('Updated {:d} "{}" documents with default value "{!s}" for property "{!s}"'.format(
            updated_count, collection, default, field
        ))

def update_database():
    logger.info('Updating Database')
    # ensure all beams have energy and particle type
    db_field_ensure_exists(DBCOLLECTIONS.BEAMPHOTON, 'particletype', PARTICLETYPE.PHOTON)
    db_field_ensure_exists(DBCOLLECTIONS.BEAMPHOTON, 'energy', '6MV', filter={'particletype': PARTICLETYPE.PHOTON})
    db_field_ensure_exists(DBCOLLECTIONS.BEAMPHOTON, 'energy', 10, filter={'particletype': PARTICLETYPE.ELECTRON})
    db_field_ensure_exists(DBCOLLECTIONS.BEAMPHOTON, 'gps_template', None)
    db_field_ensure_exists(DBCOLLECTIONS.BEAMPHOTON, 'ptv_margin', None)


    # ensure all beam documents have angle properties
    db_field_ensure_exists(DBCOLLECTIONS.BEAMPHOTON, 'angle_couch', 0)
    db_field_ensure_exists(DBCOLLECTIONS.BEAMPHOTON, 'angle_coll', 0)

    # ensure all sim documents have storage_type and sparse_threshold props
    db_field_ensure_exists(DBCOLLECTIONS.SIMULATION, 'storage_type', STORAGETYPE.DENSE)
    db_field_ensure_exists(DBCOLLECTIONS.SIMULATION, 'sparse_threshold', 0.0)

    # additional MC callargs
    db_field_ensure_exists(DBCOLLECTIONS.SIMULATION, 'callargs', [])
    db_field_ensure_exists(DBCOLLECTIONS.SIMULATION, 'priority', None)

    # mag field
    db_field_ensure_exists(DBCOLLECTIONS.SIMULATION, 'magnetic_field', (0,0,0,'tesla'))

    # tag field
    db_field_ensure_exists(DBCOLLECTIONS.SIMULATION, 'tag', None)

#====================================
# HELPERS
#====================================
def objid2str(doc, inplace=False):
    """search through dict/doc and replace all ObjectId types with strings"""
    if inplace:
        newdoc = doc
    else:
        newdoc = copy.copy(doc)
    if isinstance(newdoc, dict):
        for k, v in newdoc.items():
            newdoc[k] = objid2str(v, inplace)
    elif isinstance(newdoc, (list, tuple)):
        for ii, v in enumerate(newdoc):
            newdoc[ii] = objid2str(v, inplace)
    elif isinstance(newdoc, ObjectId):
        newdoc = str(newdoc)
    return newdoc

def dbrelpath(path):
    """create shortened path relative to dataroot for relocatability of data-store"""
    if isinstance(path, str):
        return os.path.relpath(path, DATASTORE.DATAROOT)
    elif isinstance(path, (list, tuple)):
        return [os.path.relpath(p, DATASTORE.DATAROOT) for p in path]
    elif path is None:
        return None
    else:
        raise TypeError('path must be a string or iterable of strings')

def dbabspath(path):
    """expand relative path to absolute path using current value of dataroot"""
    if isinstance(path, str):
        return os.path.abspath(pjoin(DATASTORE.DATAROOT, path))
    elif isinstance(path, (list, tuple)):
        return [os.path.abspath(pjoin(DATASTORE.DATAROOT, p)) for p in path]
    elif path is None:
        return None
    else:
        raise TypeError('path must be a string or iterable of strings')

def make_json_friendly(doc):
    """search through dict/doc and replace non jsonofyable objects with strings"""
    newdoc = copy.copy(doc)
    if isinstance(newdoc, dict):
        for k, v in newdoc.items():
            newdoc[k] = make_json_friendly(v)
    elif isinstance(newdoc, (list, tuple)):
        for ii, v in enumerate(newdoc):
            newdoc[ii] = make_json_friendly(v)
    else:
        try:
            json.dumps(newdoc)
        except:
            newdoc = str(newdoc)
    return newdoc

def check_sim_ready(sim_id):
    """check that everything is ready to go before trying to process a simulation job"""
    try:
        simdoc = get_doc(DBCOLLECTIONS.SIMULATION, sim_id)
        geom_doc = get_doc(DBCOLLECTIONS.MCGEOM, simdoc['geom_id'])
        if geom_doc['procstatus']['status'] == PROCSTATUS.FINISHED and geom_doc['geomfile'] is not None:
            return True
    except Exception as e:
        logger.exception('An error occured while checking the readyness of simulation "{}"'.format(sim_id))
    return False

def pack_sim_filedata(simdoc):
    for jj in range(len(simdoc['samples'])):
        for fkey in ['dosefile', 'densfile', 'fluencefile']:
            fp = dbabspath(simdoc['samples'][jj][fkey])
            fdesc = socketio.pack_file_binary(fp, name=fkey)
            simdoc['samples'][jj][fkey] = fdesc
    return simdoc

def doi_exists(doi):
    return (db[DBCOLLECTIONS.IMAGES].find_one({'doi': doi}) is not None)

def ctrecord_exists(ct_seriesid):
    q = db[DBCOLLECTIONS.IMAGES].find_one({'uid': ct_seriesid})
    return (q is not None)

def update_doc_status(collection, id, status, message=None, time_elapsed=None):
    db[collection].update_one(
        {'_id': ObjectId(id)},
        {'$set': {'procstatus.status': status, 'procstatus.date_modified': datetime.now(),
                  'procstatus.message': message, 'procstatus.time_elapsed': time_elapsed}}
    )

def increment_refcount(collection, id, n=1, field='refcount', update=None, array_filters=None):
    """add 1 (optionally 'n') to the 'refcount' field of the specified document
    negative values of n are allowed to decrement instead"""
    if update is None:
        update = {'$inc': {field: n}}
    res = db[collection].update_one(filter={'_id': ObjectId(id)},
                                    update=update,
                                    array_filters=array_filters,
                                    )
    assert res.modified_count == 1

def increment_refcount_structure(image_id, structure_id, n=1):
    """modify refcount spefically for structure subdoc in image document at same time as image ref"""
    update = {'$inc': {'refcount': n, 'structures.$[structure].refcount': n}}
    array_filters = [{'structure._id': ObjectId(structure_id)}, ]
    increment_refcount(DBCOLLECTIONS.IMAGES, image_id, n=n, update=update, array_filters=array_filters)

def get_image_doc(id=None, uid=None, doi=None):
    doc = None
    if id is not None:
        doc = db[DBCOLLECTIONS.IMAGES].find_one({'_id': ObjectId(id)})
    if doc is None and uid is not None:
        docs = list(db[DBCOLLECTIONS.IMAGES].find({'uid': uid}))
        if len(docs) > 1:
            raise RuntimeError('More than one image doc was matched using the uid "{}"' \
                               ' for lookup. Please select one and try again using image_id instead'.format(uid))
        if len(docs):
            doc = docs[0]
    if doc is None and doi is not None:
        docs = list(db[DBCOLLECTIONS.IMAGES].find({'doi': doi}))
        if len(docs) > 1:
            raise RuntimeError('More than one image doc was matched using the doi "{}"' \
                               ' for lookup. Please select one and try again using image_id instead'.format(doi))
        if len(docs):
            doc = docs[0]
    if doc is None:
        raise ValueError("Couldn't locate the requested document")
    return doc

def get_geometry_doc(id=None, image_id=None):
    if id is None:
        geomdocs = list(db[DBCOLLECTIONS.MCGEOM].find({'image_id': ObjectId(image_id)}))
        if len(geomdocs)>1:
            raise RuntimeError('Image has more than 1 referenced geometry ({}). Please select one and ' \
                               'retry with geom_id specified'.format(len(geomdocs)))
        geomdoc = geomdocs[0]
    else:
        geomdoc = db[DBCOLLECTIONS.MCGEOM].find_one({'_id': ObjectId(id)})
    return geomdoc

def get_docs_by_status(collection, status, **kwargs):
    return db[collection].find({'procstatus.status': status}, **kwargs)

def get_corrupted_simulation_docs():
    """Returns iterable of simdocs without samples (when num_paticles >0)

    When num_particle<=0, no samples are created and the simulation is considered Finished anyway.
    These ignored simulations
    """
    return db[DBCOLLECTIONS.SIMULATION].find({
        'procstatus.status': PROCSTATUS.FINISHED,
        'samples': {'$exists': True, "$eq": []},
        'num_particles': {'$gt': 0},
    })

def get_doc(collection, id):
    """returns db document for specified id from specified collection"""
    return make_json_friendly(db[collection].find_one({'_id': ObjectId(id)}))

def get_ctvolume(image_id=None, imagedoc=None):
    """create 3d numpy array for specified CT db object given its objectid"""
    if image_id is None and imagedoc is None:
        raise ValueError('Must specify at least one function argument')
    if imagedoc is None:
        imagedoc = get_doc(DBCOLLECTIONS.IMAGES, image_id)
    ctfiles = dbabspath(imagedoc['images'])
    return dicomutils.extract_voxel_data(ctfiles)

def get_structure_mask(image_id, structure_id, margin=None):
    """Load or generate a structure mask array

    mask file is guaranteed to exist since it is always created in structure_insert()
    ** No margin is ever saved into the mask file. It needs to be added every time if requested
    """
    imagedoc = get_doc(DBCOLLECTIONS.IMAGES, image_id)
    structuredoc = next((x for x in imagedoc['structures'] if x['_id']==str(structure_id)), None)
    if structuredoc is None:
        raise ValueError("valid structure document couldn't be located")

    frame = FrameOfReference()
    frame.start = imagedoc['coordsys']['start']
    frame.size = imagedoc['coordsys']['size']
    frame.spacing = imagedoc['coordsys']['spacing']
    mask = np.load(dbabspath(structuredoc['maskfile']))

    if margin is not None and margin>0:
        maskvol = Volume.fromArray(mask, frame)
        mask = binary_expansion(maskvol, radius=margin, inplace=True).data

    return mask

def find_boundbox_from_mask(mask, frame, buffer=0):
    """Get the tight bounding box around all non-zero elements of mask array
    optional buffer in mm is allowed and validated (clipped) against the full size of the given frame"""
    z = np.any(mask, axis=(1, 2))
    y = np.any(mask, axis=(0, 2))
    x = np.any(mask, axis=(0, 1))

    zmin, zmax = np.where(z)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    xmin, xmax = np.where(x)[0][[0, -1]]

    bbox = FrameOfReference(
        start=[frame.start[ii]+v*frame.spacing[ii]-buffer for (ii, v) in enumerate([xmin, ymin, zmin])],
        size=[int(math.ceil(l-f+2*buffer/frame.spacing[ii])) for (ii, (f, l)) in enumerate([(xmin, xmax), (ymin, ymax), (zmin, zmax)])],
        spacing=frame.spacing,
    )
    return dicomutils.validate_bbox(bbox, frame)

def get_geometry_volume(geom_id):
    """reconstruct geometry slice from database doc"""
    # get geomdoc
    geomdoc = get_doc(DBCOLLECTIONS.MCGEOM, geom_id)
    imagedoc = get_doc(DBCOLLECTIONS.IMAGES, geomdoc['image_id'])
    ics = imagedoc['coordsys']
    gcs = geomdoc['coordsys']
    if ics['spacing'] != gcs['spacing']:
        logger.info('geometry voxel spacing {} differs from image voxel spacing {}'.format(
            gcs['spacing'], ics['spacing']
        ))

    # load volume and get subvolume defined by geometry doc's coordsys
    volume, _ = get_ctvolume(geomdoc['image_id'])
    volume = Volume.fromArray(volume, frameofreference=FrameOfReference(**ics))
    logger.debug('image vol = {} , shape:{}'.format(volume.frameofreference, volume.data.shape))
    subvol = volume.conformTo(FrameOfReference(**gcs))
    logger.debug('geometry subvol = {} , shape:{}'.format(subvol.frameofreference, subvol.data.shape))
    return subvol.data, gcs['spacing']

def get_number_of_docs_by_status(collection, status):
    agg = db[collection].aggregate([
        {'$match': {'procstatus.status': status}},
        {'$count': 'count'}
    ])
    try: count = agg.next()['count']
    except StopIteration as e: count = 0
    return count

def get_number_of_docs_for_image_by_status(image_id, status_list, num_particles=None):
    if not isinstance(status_list, list):
        status_list = [status_list]

    filter = {'image_id': ObjectId(image_id),
              'procstatus.status': {'$in': status_list}}
    if num_particles is not None:
        filter['num_particles'] = num_particles
    agg = db[DBCOLLECTIONS.SIMULATION].aggregate([
        {'$match': filter},
        {'$count': 'count'}
    ])
    try: count = agg.next()['count']
    except StopIteration as e: count = 0
    return count

#====================================
# DOCUMENT TEMPLATES
#====================================
def gen_doc_coordsys(start, size, spacing):
    """define coordinate system"""
    doc = {
        'start':   start,   # float[3] (unit: mm)
        'size':    size,    # int[3]
        'spacing': spacing, # float[3] (unit: mm)
    }
    return doc

def gen_doc_procstatus(status=PROCSTATUS.PENDING, date_modified=None, message=None, time_elapsed=None):
    """general processing status"""
    doc = {
        'status':        status,
        'date_modified': date_modified,
        'message':       message,
        'time_elapsed':  time_elapsed,
    }
    return doc

def gen_doc_structure(name, centroid, boundbox, maskfile=None, objectid=None):
    doc = {
        '_id':        ObjectId(objectid) if objectid is not None else ObjectId(),
        'name':       name,
        'date_added': datetime.now(),
        'maskfile':   dbrelpath(maskfile),
        'centroid':   centroid,
        'boundbox':   boundbox,
        'refcount':   0,
    }
    return doc


def gen_doc_image(ctid, doi, ctfiles, rtstructfile, coordsys, structures=None, objectid=None):
    doc = {
        '_id':        ObjectId(objectid) if objectid is not None else ObjectId(),
        'uid':        ctid,
        'date_added': datetime.now(),
        'doi':        doi,
        'images':     dbrelpath(ctfiles),
        'rtstruct':   dbrelpath(rtstructfile),
        'coordsys':   coordsys,
        'structures': structures if structures is not None else [],
        'refcount':   0,
    }
    return doc

def gen_doc_photonbeam(image_id, geom_id, structure_id, mlrole, angle_gantry, angle_couch, angle_coll, isocenter, sad, fmapdims, beamletspacing, beamletsize, ptv_margin, beamlets=None, particletype=PARTICLETYPE.PHOTON, energy='6MV', gps_template=None, objectid=None):
    doc = {
        '_id':            ObjectId(objectid) if objectid is not None else ObjectId(),
        'date_added':     datetime.now(),
        'image_id':       ObjectId(image_id),
        'geom_id':        ObjectId(geom_id),
        'structure_id':   ObjectId(structure_id),
        'mlrole':         mlrole,
        'angle_gantry':   angle_gantry,
        'angle_couch':    angle_couch,
        'angle_coll':     angle_coll,
        'sad':            sad,
        'fmapdims':       fmapdims,
        'beamletspacing': beamletspacing,
        'beamletsize':    beamletsize,
        'isocenter':      isocenter,
        'energy':         energy,
        'beamlets':       beamlets if beamlets is not None else [],
        'ptv_margin':     ptv_margin,
        'particletype':   particletype,
        'gps_template':   gps_template,
    }
    return doc

def gen_doc_protonbeam(image_id, geom_id, structure_id, mlrole, angle_gantry, angle_couch, angle_coll, isocenter, sad, fmapdims, beamletspacing, beamletsize, ptv_margin, beamlets=None, particletype=PARTICLETYPE.PHOTON, energy='6MV', gps_template=None, objectid=None):
    doc = {
        '_id':            ObjectId(objectid) if objectid is not None else ObjectId(),
        'date_added':     datetime.now(),
        'image_id':       ObjectId(image_id),
        'geom_id':        ObjectId(geom_id),
        'structure_id':   ObjectId(structure_id),
        'mlrole':         mlrole,
        'angle_gantry':   angle_gantry,
        'angle_couch':    angle_couch,
        'angle_coll':     angle_coll,
        'sad':            sad,
        'fmapdims':       fmapdims,
        'beamletspacing': beamletspacing,
        'beamletsize':    beamletsize,
        'isocenter':      isocenter,
        'energy':         energy,
        'beamlets':       beamlets if beamlets is not None else [],
        'ptv_margin':     ptv_margin,
        'particletype':   particletype,
        'gps_template':   gps_template,
    }
    return doc

def gen_doc_mcgeometry(image_id, coordsys, geomfile=None, procstatus=None, geomtype=MCGEOTYPE.BULKDENS, objectid=None):
    doc = {
        '_id':        ObjectId(objectid) if objectid is not None else ObjectId(),
        'date_added': datetime.now(),
        'procstatus': procstatus if procstatus is not None else gen_doc_procstatus(),
        'image_id':   ObjectId(image_id),
        'geomfile':   dbrelpath(geomfile),
        'geomtype':   geomtype,
        'coordsys':   coordsys,
        'refcount':   0,
    }
    return doc

def gen_doc_beamlet(position, simulations=None, gpsfile=None, objectid=None):
    """used by beamlets/spots"""
    doc = {
        '_id':         ObjectId(objectid) if objectid is not None else ObjectId(),
        'date_added':  datetime.now(),
        'position':    position, # int[2]
        'gpsfile':     dbrelpath(gpsfile),
        'simulations': simulations if simulations is not None else [],
    }
    return doc

def gen_doc_spot(position, energy, simulations=None, gpsfile=None, objectid=None):
    """used by beamlets/spots"""
    doc = {
        '_id':         ObjectId(objectid) if objectid is not None else ObjectId(),
        'date_added':  datetime.now(),
        'position':    position, # int[2]
  'energy':      energy,
        'gpsfile':     dbrelpath(gpsfile),
        'simulations': simulations if simulations is not None else [],
    }
    return doc

def gen_doc_simulation(image_id, geom_id, beam_id, subbeam_id, vartype, num_runs, num_particles, magnetic_field=None, storage_type=STORAGETYPE.DENSE, sparse_threshold=0.0, callargs=None, samples=None, procstatus=None, priority=None, tag=None, objectid=None):
    doc = {
        '_id':            ObjectId(objectid) if objectid is not None else ObjectId(),
        'date_added':     datetime.now(),
        'procstatus':     procstatus if procstatus is not None else gen_doc_procstatus(),
        'image_id':       ObjectId(image_id),
        'geom_id':        ObjectId(geom_id),
        'beam_id':        ObjectId(beam_id),
        'subbeam_id':     ObjectId(subbeam_id),
        'vartype':        vartype,
        'num_runs':       num_runs,
        'num_particles':  num_particles,
        'magnetic_field': magnetic_field,
        'storage_type':   storage_type,
        'sparse_threshold': sparse_threshold,
        'callargs':       callargs if callargs is not None else [],
        'priority':       priority,
        'tag':            tag,  # Optional string field for filtering results
        'samples':        samples if samples is not None else [],
    }
    return doc

def gen_doc_sample(dosefile, densfile, fluencefile, objectid=None):
    """instance of a simulation run's output. This is different from doc_simulation since multiple independent
    runs can be generated in a single execution of the simulation program for efficiency"""
    doc = {
        '_id':         ObjectId(objectid) if objectid is not None else ObjectId(),
        'dosefile':    dbrelpath(dosefile),
        'densfile':    dbrelpath(densfile),
        'fluencefile': dbrelpath(fluencefile),
    }
    return doc
