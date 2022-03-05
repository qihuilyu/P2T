import os
from os.path import join as pjoin
import tempfile
import re
import shutil

import numpy as np
import numpy.random

import payloadtypes
import socketio
from database import ObjectId
import database
import dicomutils
from api_enums import (MESSAGETYPE, STATUS, MLROLE, PROCSTATUS,
                       VARTYPE, MCGEOTYPE, DBCOLLECTIONS)
from rttypes.frame import FrameOfReference
import generate_input
import geometry
import log
logger = log.get_module_logger(__name__)

#========
# HELPERS
#========
def cursor_size(cursor):
    """Get the size of the cursor"""
    count = 0
    while cursor.alive:
        count += 1
        cursor.next()
    cursor.rewind()
    return count

def get_beamdoc_pipeline(beam_id, beamfield=''):
    """Modular recursive lookup beginning from beam level"""
    if beamfield:
        beamfield += '.'
    pipeline = [
        {'$match': {beamfield+'_id': ObjectId(beam_id)}},
        {'$unwind': '$beamlets'},
        {'$lookup': {'from': DBCOLLECTIONS.SIMULATION,
                     'localField': beamfield+'beamlets.simulations',
                     'foreignField': '_id',
                     'as': beamfield+'beamlets.simulations'}},
        {'$group': {'_id': '$'+beamfield+'_id',
                    beamfield+'beamlets': {'$push': '$'+beamfield+'beamlets'}}}
    ]
    return pipeline

def find_beam_doc(beam_id, recursive=False):
    if recursive == True:
        beamdoc = database.db[DBCOLLECTIONS.BEAMPHOTON].aggregate([
            *get_beamdoc_pipeline(beam_id),
        ]).next()
    else:
        beamdoc = database.get_doc(DBCOLLECTIONS.BEAMPHOTON, beam_id)
    if beamdoc is None:
        raise ValueError('No valid beam document could be located for id "{}"'.format(beam_id))
    return beamdoc

def find_image_doc(payload: payloadtypes.RestReqImageBase):
    return database.get_image_doc(id=payload.image_id,
                                  uid=payload.image_uid,
                                  doi=payload.image_doi)

#=============
# REST ACTIONS
#=============
def rest_image_insert(payload: payloadtypes.RESTReqImageInsert):
    """create a single CT record return id"""
    ctfiles = payload.files['ct']
    rtstructfile = payload.files['rtstruct']
    image_id = database.image_insert(ctfiles, rtstructfile, payload.doi)
    response = {
        'id': str(image_id),
    }
    return response

def rest_image_get(payload: payloadtypes.RESTReqImageGet):
    return find_image_doc(payload)

def rest_image_delete(payload: payloadtypes.RESTReqImageDelete):
    # first locate image
    doc = find_image_doc(payload)
    # delete image and all dependent documents
    database.image_delete(doc['_id'])
    return {'id': str(doc['_id'])}

def rest_structure_insert(payload: payloadtypes.RESTReqStructureInsert):
    if payload.mask is not None:
        mask = np.frombuffer(socketio.unpack_file_to_memory(payload.mask), dtype=payload.mask['dtype'])
        mask = mask.reshape(payload.mask['size'])
    else:
        mask = None

    structure_id = database.structure_insert(
        image_id=payload.image_id,
        name=payload.name,
        mask=mask,
    )

    response = {
        'id': str(structure_id),
    }
    return response

def rest_structure_get(payload: payloadtypes.RESTReqStructureGet):
    doc = database.db[DBCOLLECTIONS.IMAGES].aggregate([
        {'$match': {'_id': ObjectId(payload.image_id)}},
        {'$limit': 1},
        {'$unwind': '$structures'},
        {'$match': {'$or': [{'structures._id': ObjectId(payload.structure_id)}, {'structures.name': payload.structure_name}]}},
    ]).next()
    if doc is None:
        raise ValueError("Couldn't locate the requested document")
    response = doc['structures']
    return response

def rest_structure_delete(payload: payloadtypes.RESTReqStructureDelete):
    database.structure_delete(payload.image_id, payload.structure_id)
    response = {
        'id': str(payload.structure_id)
    }
    return response

def rest_geometry_insert(payload: payloadtypes.RESTReqGeometryInsert):
    # confirm ct record exists
    imagedoc = database.get_doc(DBCOLLECTIONS.IMAGES, payload.image_id)
    if imagedoc is None:
        raise ValueError("Couldn't locate valid image document")

    # inherit unspecified coordsys params
    coordsys = {}
    for k in ('start', 'size', 'spacing'):
        if payload.coordsys.get(k, None) is None:
            coordsys[k] = imagedoc['coordsys'][k]
            logger.info('no geometry coordys "{0}" provided. Using image coordsys "{0}" by default'.format(k))
        else:
            coordsys[k] = payload.coordsys.get(k)

    # generate and insert new record
    geom_objectid = database.geometry_insert(
        image_id=payload.image_id,
        **coordsys,
        geomtype=payload.geomtype,
    )
    return {'id': str(geom_objectid)}

def rest_geometry_get(payload: payloadtypes.RESTReqGeometryGet):
    if payload.filter:
        doc = list(database.db[DBCOLLECTIONS.MCGEOM].find(filter=payload.filter))
    else:
        doc = database.get_doc(DBCOLLECTIONS.MCGEOM, payload.geom_id)
    if doc is None:
        raise ValueError("Couldn't locate the requested document")
    return doc

def rest_geometry_delete(payload: payloadtypes.RESTReqGeometryDelete):
    database.geometry_delete(payload.geom_id)
    return {'id': payload.geom_id}


def rest_beamphoton_get(payload: payloadtypes.RESTReqBeamPhotonGet):
    # get beam doc
    if payload.filter:
        beamdoc = list(database.db[DBCOLLECTIONS.BEAMPHOTON].find(filter=payload.filter))
    else:
        beamdoc = find_beam_doc(payload.beam_id, recursive=payload.recursive)
    return beamdoc

def rest_beamphoton_insert(payload: payloadtypes.RESTReqBeamPhotonInsert):
    # generate and insert a beam doc
    beam_ids = []
    for beam in payload.beams:
        beam_objectid = database.beam_insert(
            geom_id=payload.geom_id,
            structure_id=payload.structure_id,
            angle_gantry=beam.angle_gantry,
            angle_couch=beam.angle_couch,
            angle_coll=beam.angle_coll,
            sad=beam.sad,
            fmapdims=beam.fmapdims,
            beamletspacing=beam.beamletspacing,
            beamletsize=beam.beamletsize,
            particletype=beam.particletype,
            energy=beam.energy,
            isocenter=beam.isocenter,
            beamlets=beam.beamlets,
            ptv_margin=beam.ptv_margin,
            mlrole=payload.mlrole,
            gps_template=beam.gps_template,
        )
        beam_ids.append(beam_objectid)

    response = {
        'ids': [str(x) for x in beam_ids]
    }
    return response

def rest_beamphoton_delete(payload: payloadtypes.RESTReqBeamPhotonDelete):
    # get beamdoc
    beamdoc = find_beam_doc(payload.beam_id)

    # delete all associated simulations
    simdocs = database.db[DBCOLLECTIONS.SIMULATION].find({'beam_id': ObjectId(beamdoc['_id'])})
    for simdoc in simdocs:
        database.db[DBCOLLECTIONS.SIMULATION].delete_one({'_id': simdoc['_id']})
        #  logger.debug(pjoin(database.dbabspath(database.DATASTORE.SIMDATA), str(simdoc['_id'])))
        shutil.rmtree(pjoin(database.dbabspath(database.DATASTORE.SIMDATA), str(simdoc['_id'])), ignore_errors=True)
        logger.debug('Deleted simulation "{}"'.format(simdoc['_id']))

    database.db[DBCOLLECTIONS.BEAMPHOTON].delete_one({'_id': ObjectId(beamdoc['_id'])})
    return {'id': str(beamdoc['_id'])}


def rest_simulation_insert_photon(payload: payloadtypes.RESTReqSimulationInsert):
    beamdoc = find_beam_doc(payload.beam_id)
    image_id = beamdoc['image_id']
    geom_id = beamdoc['geom_id']

    selected_beamlets = []
    if payload.beamlets is None:
        selected_beamlets = [b['_id'] for b in beamdoc['beamlets']]
    elif isinstance(payload.beamlets, dict):
        if 'ids' in payload.beamlets:
            valid_ids = [ObjectId(b['_id']) for b in beamdoc['beamlets']]
            for beamlet_id in payload.beamlets['ids']:
                beamlet_id = ObjectId(beamlet_id)
                if not beamlet_id in valid_ids:
                    raise ValueError('beamlet "{!s}" not found in beam "{!s}"'.format(beamlet_id, beamdoc['_id']))
                selected_beamlets.append(beamlet_id)
        elif 'positions' in payload.beamlets:
            valid_positions = {tuple(b['position']): ObjectId(b['_id']) for b in beamdoc['beamlets']}
            for beamlet_pos in payload.beamlets['positions']:
                if not tuple(beamlet_pos) in valid_positions:
                    raise ValueError('beamlet at position "{!s}" not found in beam "{!s}"'.format(beamlet_pos, beamdoc['_id']))
                selected_beamlets.append(valid_positions[beamlet_pos])
        elif 'random-count' in payload.beamlets:
            selected_beamlets = [ObjectId(beamdoc['beamlets'][ii]['_id']) for ii in
                                 np.random.choice(len(beamdoc.beamlets),
                                                  size=payload.beamlets['random-count'],
                                                  replace=False)]
    if not len(selected_beamlets):
        raise ValueError('Simulations couldn\'t be inserted because no beamlets were selected')

    sim_ids = database.simulation_insert_bundled(beam_id=payload.beam_id,
                                                 subbeam_ids=selected_beamlets,
                                                 vartype=payload.vartype,
                                                 num_runs=payload.num_runs,
                                                 num_particles=payload.num_particles,
                                                 magnetic_field=payload.magnetic_field,
                                                 storage_type=payload.storage_type,
                                                 sparse_threshold=payload.sparse_threshold,
                                                 callargs=payload.callargs,
                                                 priority=payload.priority,
                                                 tag=payload.tag)

    response = {'ids': [str(x) for x in sim_ids]}
    return response

def rest_simulation_get(payload: payloadtypes.RESTReqSimulationGet):
    if payload.filter:
        simdoc = list(database.db[DBCOLLECTIONS.SIMULATION].find(filter=payload.filter))
        if payload.filedata:
            # pack file data into docs
            for ii in range(len(simdoc)):
                simdoc[ii] = database.pack_sim_filedata(simdoc[ii])
    else:
        simdoc = database.get_doc(DBCOLLECTIONS.SIMULATION, payload.sim_id)
        if payload.filedata:
            simdoc = database.pack_sim_filedata(simdoc)
    return simdoc

def rest_simulation_insert(payload: payloadtypes.RESTReqSimulationInsert):
    return rest_simulation_insert_photon(payload)

def rest_simulation_delete(payload: payloadtypes.RESTReqSimulationDelete):
    if not payload.filter:
        raise ValueError("Database filter was not provided in RESTReqSimulationDelete. Nothing will be done.")

    simdocs = list(database.db[DBCOLLECTIONS.SIMULATION].find(filter=payload.filter))

    deleted_sim_ids = []
    for simdoc in simdocs:
        database.simulation_delete(simdoc['_id'], remove_refs=True)
        deleted_sim_ids.append(str(simdoc['_id']))

    logger.debug('Deleted {} simulation docs in response to SimulationDelete REST request with filter {!s}'.format(
        len(deleted_sim_ids), payload.filter))
    return {'deleted_ids': deleted_sim_ids}

def rest_sample_get(payload: payloadtypes.RESTReqSampleGet):
    # lookup simdoc
    simdoc = database.db[DBCOLLECTIONS.SIMULATION].aggregate([
        {'$match': {'_id': ObjectId(payload.sim_id)}},
        {'$unwind': '$samples'},
        {'$match': {'samples._id': ObjectId(payload.sample_id)}}
    ]).next()
    return simdoc

def rest_sample_delete(payload: payloadtypes.RESTReqSampleDelete):
    # lookup simdoc
    update = {'$pull': {'samples': {'_id': ObjectId(payload.sample_id)}}}
    if payload.decrement_nruns:
        update['$inc'] = {'num_runs': -1}
    oldsimdoc = database.db[DBCOLLECTIONS.SIMULATION].find_one_and_update(
        filter={'_id': ObjectId(payload.sim_id)},
        update=update,
    )
    if oldsimdoc is None:
        raise RuntimeError('Problem while deleting simulation sample "{}::{}"'.format(payload.sim_id, payload.sample_id))

    # remove files
    treepath = pjoin(database.dbabspath(database.DATASTORE.SIMDATA), str(oldsimdoc['beam_id']), str(oldsimdoc['subbeam_id']), str(payload.sim_id), str(payload.sample_id))
    logger.debug('Removing sample data from "{}"'.format(treepath))
    shutil.rmtree(treepath, ignore_errors=True)

    response = {'sim_id': ObjectId(payload.sim_id), 'sample_id': ObjectId(payload.sample_id)}
    return response
