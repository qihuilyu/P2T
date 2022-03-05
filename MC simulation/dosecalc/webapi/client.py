#!/usr/bin/env python3

import sys, os
from os.path import join as pjoin
import shutil
import signal
import socket
import time
import copy
import random
import socket
import math
import json
import logging
import tempfile
import multiprocessing
import traceback
from datetime import datetime, timedelta
from pprint import pprint, pformat
from collections import namedtuple

from rttypes.frame import FrameOfReference
import numpy as np
import numpy.random
import pymongo
from bson.objectid import ObjectId
import h5py
try:
    from tqdm import tqdm
except ImportError as e:
    # return no-op wrapper
    def tqdm(iterable, *args, **kwargs):
        return iterable

import payloadtypes
import defaults
from api_enums import (MESSAGETYPE, STATUS, MLROLE, PROCSTATUS, MCGEOTYPE,
                       VARTYPE, DBCOLLECTIONS, PARTICLETYPE,
                       STORAGETYPE)
from commandmenu import CommandMenuBase, menucommand
import parse
import socketio
import dicomutils
from loaders import ArrayLoader
from sparse import SparseMatrixCOO
from utils import load_bin, save_bin, none_or_type, get_resizing_params
import database
from filewriters import RotatingNPFileWriter
import quan_config
import restactions
from beamlist import read_beamlist, ensure_exclusive_setting
import log
logger = log.get_module_logger("client")

LOW_PARTICLES  = int(18e6)
HIGH_PARTICLES = int(2e3)
dsaddr = None


def get_simdoc(filter):
    """Multiprocessing worker function"""
    database.reinit_dbclient()
    return database.db[DBCOLLECTIONS.SIMULATION].find_one(filter)

def sort_beamlets_columnorder(beamdoc):
    return sorted(beamdoc['beamlets'],
                  key=lambda x: x['position'][0]*beamdoc['fmapdims'][0]+x['position'][1])

def validate_rest_response(response):
    if response['status'] != STATUS.SUCCESS:
        raise RuntimeError('action failed with error: {!s}'.format(response['message']))
    return response['content']

def load_bbox(bbox_dict, voxelsize):
    """load bbox coordsys from bbox.json file"""
    bbox = payloadtypes.CoordSys()
    bbox.spacing = voxelsize
    bbox.start = [bbox_dict[k][0] for k in ['x','y','z']]
    bbox.size = [int(math.ceil((bbox_dict[k][1]-bbox.start[ii])/voxelsize[ii])) for ii, k in enumerate(['x','y','z'])]
    return bbox

#==================
# Command Line Menu
#==================
class CMenuDebug(CommandMenuBase):
    description = 'Client for interacting with distributed Monte Carlo Dose Calculation network'

    @staticmethod
    def generate_simulation_payload(geometry_file, gps_file, reply_host, reply_port):
        """simulation test payload send directly to computeserver"""
        payload = payloadtypes.SimInstruction()
        payload.id = str(database.ObjectId())
        payload.num_vacant_threads = 2
        payload.beam_id = str(database.ObjectId())
        payload.subbeam_id = str(database.ObjectId())
        payload.files = {
            'geometry': socketio.pack_file_text(name='mcgeo.txt', file=geometry_file),
            'gps': socketio.pack_file_text(name='gps.mac', file=gps_file),
        }
        payload.simulations = [
            payloadtypes.SimulationConfig.fromdict({'id': str(database.ObjectId()), 'num_runs': 1,
                                                    'num_particles': LOW_PARTICLES, 'vartype': VARTYPE.LOW, }),
            payloadtypes.SimulationConfig.fromdict({'id': str(database.ObjectId()), 'num_runs': 30,
                                                    'num_particles': HIGH_PARTICLES, 'vartype': VARTYPE.HIGH, }),
        ]
        payload.reply_host = reply_host
        payload.reply_port = reply_port
        return payload

    @staticmethod
    def submit_request(payload, timeout=10):
        starttime = time.perf_counter()
        ipidx = 0
        while True:
            if timeout and timeout > 0 and time.perf_counter()-starttime > timeout:
                raise OSError('Timeout reached while trying to submit processing request')
            try:
                cs_addr = defaults.cs_address[ipidx]
                response = payload.send_request(cs_addr, timeout=None, conection_timeout=1)
                if response['status'] == STATUS.SUCCESS:
                    break
            except (socket.timeout, ConnectionRefusedError, ConnectionResetError) as err:
                logger.debug('timeout while trying to connect to "{}:{}"'.format(*cs_addr))
                ipidx = (ipidx+1)%len(defaults.cs_address)
                time.sleep(1)

        return cs_addr


    @menucommand
    def test_simulation(self, parser):
        """Send a test simulation processing request"""
        parser.add_argument('num', nargs='?', type=int, default=0, help='send a number of tests immediately')
        parser.add_argument('--replyhost', '--rh', default='127.0.0.1')
        parser.add_argument('--replyport', '--rp', type=int, default=5567)
        self.args = parser.parse_args(namespace=self.args)

        if self.args.num > 0:
            for jobnum in range(self.args.num):
                logger.info('submitting simulation request {} of {}'.format(jobnum+1, self.args.num))
                CMenuDebug.submit_request( CMenuDebug.generate_simulation_payload('./test/mcgeo.txt', './test/gps.mac',
                                                                       self.args.replyhost, self.args.replyport), timeout=None )
                logger.info('request accepted')
            return

        while True:
            input('press enter to send a task request\n')
            CMenuDebug.submit_request( CMenuDebug.generate_simulation_payload('./test/mcgeo.txt', './test/gps.mac',
                                                        self.args.replyhost, self.args.replyport), timeout=None )
            logger.info('request accepted')

class CMenuDatabase(CommandMenuBase):
    description='Apply actions directly on database'

    def register_addl_args(self, parser):
        parse.register_db_args(parser)
        parser.add_argument('--data', '-d', type=str, default="db_data", help="set data root directory")

    def run_after_parse(self):
        # set global settings
        database.init_dbclient(host=self.args.dbhost, port=self.args.dbport,
                               dbname=self.args.dbname, auth=self.args.dbauth)
        database.InitDataStorage(self.args.data)

    #====================================
    # REUSABLE DB MANIPULATION
    #====================================
    @menucommand
    def resetdb(self, parser):
        """Reset specified database, clearing all data from it"""
        dbname = self.args.dbname
        client, db = database.dbclient, database.db
        logger.info("Resetting database \"{}\"".format(dbname))
        available_dbs = client.list_database_names()
        if not dbname in available_dbs:
            raise pymongo.database.ConfigurationError("Database \"{}\" does not exist. Options are: [{}]".format(dbname, ','.join(available_dbs)))
        logger.warning("Are you sure you want to reset the database \"{}\"?".format(dbname))
        resp = input("(y/[n]): ")
        if resp.lower() in ['y', 'yes']:
            logger.warning('Deleting database...')
            client.drop_database(dbname)
            logger.warning('Done')
        else:
            logger.warning('aborted')

        logger.warning("Would you like to also clear all data from referenced data directory \"{}\"?".format(database.DATASTORE.DATAROOT))
        resp = input("(y/[n]): ")
        if resp.lower() in ['y', 'yes']:
            logger.warning('Deleting referenced data...')
            try:
                logger.debug('attempting to delete: '+database.DATASTORE.DATAROOT)
                shutil.rmtree(database.DATASTORE.DATAROOT)
            except Exception as e:
                logger.exception('Error while attempting to delete directory tree "{}"'.format(database.DATASTORE.DATAROOT))
            logger.warning('Done')
        else:
            logger.warning('aborted')

    @menucommand
    def backupdb(self, parser):
        """Copy current state of database to new table"""
        parser.add_argument('--backup-name', default='{}_backup'.format(self.args.dbname))
        self.args = parser.parse_args(namespace=self.args)

        dbclient, db = database.dbclient, database.db
        backupdb = dbclient[self.args.backup_name]
        if self.args.backup_name in dbclient.list_database_names():
            logger.warning("database \"{}\" already exists. Would you like to overwrite?".format(self.args.backup_name))
            resp = input("(y/[n]): ")
            if resp.lower() in ['y', 'yes']:
                logger.warning('Deleting database...')
                dbclient.drop_database(self.args.backup_name)
                logger.warning('Done')
            else:
                logger.warning('Could not complete backup!')
                return

        logger.info('Copying database "{}" to "{}"'.format(self.args.dbname, self.args.backup_name))
        for collname in db.list_collection_names():
            logger.debug('copying collection "{}"'.format(collname))
            backupcoll = backupdb[collname]
            backupcoll.insert_many(db[collname].find())

    @menucommand
    def cleandb(self, parser):
        """Verify all simulation documents for file size/existence. Reset invalid docs for recalculation"""
        parser.add_argument('-n', '--dry-run', action='store_true', help='just print invalid entries')
        parser.add_argument('-a', '--action', choices=['all', 'corrupt_sims', 'leftovers'], default='all', help='Which action to take (default=all)')
        self.args = parser.parse_args(namespace=self.args)

        # the order of these cleanup commands matters
        if self.args.action in [ 'all', 'leftovers' ]:
            database.cleandb_remove_leftover_files(dryrun=self.args.dry_run)
        if self.args.action in [ 'all', 'corrupt_sims' ]:
            database.cleandb_reset_corrupt_sims(dryrun=self.args.dry_run)


    #====================================
    # TREATMENT PLANNING FUNCTIONS
    #====================================
    @menucommand
    def generate(self, parser):
        """>>Collection of results generation functions"""
        self.CMenuResultsGeneration()

    class CMenuResultsGeneration(CommandMenuBase):
        @staticmethod
        def _get_beam_ids(beamlistfile=None, image_id=None, image_uid=None, image_doi=None, geom_id=None):
            beam_ids = []
            if beamlistfile is not None:
                with open(beamlistfile, 'r') as fd:
                    for beam_id in fd:
                        beam_ids.append(beam_id.rstrip('\n'))
            else:
                # use all beams associated to image/geom pair
                imagedoc = database.get_image_doc(id= image_id,
                                                  uid=image_uid,
                                                  doi=image_doi)
                geomdoc = database.get_geometry_doc(id=geom_id,
                                                    image_id=imagedoc['_id'])
                assert str(geomdoc['image_id']) == str(imagedoc['_id'])
                beam_ids = next(database.db[DBCOLLECTIONS.BEAMPHOTON].aggregate([
                    {'$match': {"geom_id": ObjectId(geomdoc['_id'])}},
                    {'$group': { "_id": '0', "ids": { "$push": "$_id" } }},
                ]))['ids']
            return beam_ids

        @menucommand
        def dataset(self, parser):
            parser.add_argument('--nparticles', type=int, default=None, help='filter only samples for this number of particles')
            parser.add_argument('--limit-examples', type=int, default=float('inf'), help='max number of data example pairs to include')
            parser.add_argument('--fsize', type=float, default=1, help='max filesize in GB before splitting')
            parser.add_argument('--xcontext', type=none_or_type(int), default=20, help='number of slices included to either side of beamlet center (along x-axis)')
            parser.add_argument('--zcontext', type=none_or_type(int), default=20, help='number of slices included to either side of beamlet center (along z-axis)')
            parser.add_argument('--out', '-o', default=None, help='directory to dump training data')
            parser.parse_args(namespace=self.args)
            if self.args.out is None:
                self.args.out = 'dataset_{!s}'.format(datetime.now().strftime('%F_%T'))

            db = database.db
            dosefactor = 1e26 # multply raw MC dose arrays by this value before pre-processing
            loadarr = ArrayLoader(reorient=True, context=(self.args.xcontext, 50, self.args.zcontext),
                                         get_geom=True, get_label=True, multiproc=True,dosefactor=dosefactor)

            os.makedirs(self.args.out, exist_ok=True)
            error_logger = open(pjoin(self.args.out, 'errors.txt'), 'w')
            for role in [MLROLE.TRAIN, MLROLE.TEST]:
                if role == MLROLE.TEST:
                    limit_examples = 8000
                else:
                    limit_examples = self.args.limit_examples
                outdir = pjoin(self.args.out, role)
                os.makedirs(outdir, exist_ok=True)
                datafw = RotatingNPFileWriter(pjoin(outdir, role), max_fsize=self.args.fsize, texthead='image_id,beam_id,subbeam_id,sim_id,sample_id')

                # RANDOMLY SAMPLE A SUBSET OF SIMULATIONS
                beam_ids = [doc['_id'] for doc in db[DBCOLLECTIONS.BEAMPHOTON].find({
                    'mlrole': role,
                    #  'date_added': {"$gte": datetime(2020, 1, 5)},
                })]
                filter = {
                    "magnetic_field": [0.0, 0.0, 1.5, 'tesla'],
                    'procstatus.status': PROCSTATUS.FINISHED,
                    'vartype': VARTYPE.HIGH,
                    'beam_id': {"$in": beam_ids},
                }
                if self.args.nparticles:
                    filter['num_particles'] = int(self.args.nparticles)
                try:
                    nsims = db[DBCOLLECTIONS.SIMULATION].aggregate([
                        {"$match": filter},
                        {"$count": 'num_sims'}
                    ]).next()['num_sims']
                except StopIteration:
                    logger.info('no documents matched query')
                    continue

                logger.info('{} matching samples'.format(nsims))

                # GET UNIFORM RANDOM SAMPLE OF DOCS
                simdocs = random.sample(list(db[DBCOLLECTIONS.SIMULATION].find(filter)), k=min(nsims, limit_examples))
                logger.info('randomly sampled {} documents'.format(len(simdocs)))

                with multiprocessing.Pool() as pool:
                    iresults = pool.imap(loadarr, simdocs, chunksize=8)
                    for ii, (result, simdoc) in enumerate(zip(tqdm(iresults, total=len(simdocs), desc='Constructing {} dataset'.format(role.title())), simdocs)):
                        if result is None:
                            raise RuntimeError('Error while loading beamlet dose arrays for simulation "{!s}"'.format(simdoc['_id']))
                        input_arrs, geom_arr, label_arr = result[0]

                        # Check for empty dose (bad beamlet specification during random select?)
                        ctr_idx = np.array(label_arr.shape[0])//2
                        sumslice = slice(max(0, ctr_idx-5), min(label_arr.shape[2], ctr_idx+5))
                        volsum = np.sum(label_arr[sumslice, :, sumslice])
                        if volsum < 1000:
                            error_logger.write("corrupt dose ({}) on volume: {!s}\n".format(volsum, simdoc['_id']))
                            error_logger.flush()
                            continue
                        try:
                            assert np.amin(label_arr)>=0.0
                            for input_arr in input_arrs:
                                assert np.amin(input_arr)>=0.0
                        except AssertionError as e:
                            print(simdoc['_id'], np.amin(label_arr), [np.amin(input_arr) for input_arr in input_arrs])
                            raise

                        channel_arrs = []
                        for input_arr in input_arrs:
                            channel_arrs.append( np.stack([label_arr, input_arr, geom_arr], axis=-1).astype(np.float32) )
                        sample_arr = np.stack(channel_arrs, axis=0)

                        text_out = []
                        for sampledoc in simdoc['samples']:
                            text_out.append('{!s},{!s},{!s},{!s},{!s}'.format(
                                simdoc['image_id'],
                                simdoc['beam_id'],
                                simdoc['subbeam_id'],
                                simdoc['_id'],
                                sampledoc['_id'],
                            ))
                        datafw.write(sample_arr, text=text_out)
                    datafw.save()
                with open(pjoin(self.args.out, 'stats.json'), 'w') as fd:
                    json.dump({'factor': [dosefactor, dosefactor, 1.0]}, fd)

            error_logger.close()

        @menucommand
        def masks(self, parser):
            """Export a file containing mask volumes matching the image coordinate system, respecting the
            user-specified voxelsize (if it was manually defined)"""
            parser.add_argument('--image_id', type=str, help="image database id")
            parser.add_argument('--image_uid', type=str, help="image dicom uid")
            parser.add_argument('--image_doi', type=str, help="image doi")
            parser.add_argument('--geom_id', type=str, help="geometry database id")
            parser.add_argument('--numpy', action='store_true', help='also produce a .npy file for each mask')
            parser.add_argument('--out', default='masks.h5', help='file to write mask data')
            self.args = parser.parse_args()

            imagedoc = database.get_image_doc(id=self.args.image_id,
                                              uid=self.args.image_uid,
                                              doi=self.args.image_doi)
            assert imagedoc is not None
            assert 'rtstruct' in imagedoc
            geomdoc = database.get_geometry_doc(id=self.args.geom_id,
                                                image_id=imagedoc['_id'])

            # calculate resizing params
            ic = imagedoc['coordsys']

            # Get list of structures
            existing_structures = {doc['name']: doc['_id'] for doc in imagedoc['structures']}
            structure_names = set(existing_structures.keys())
            if imagedoc['rtstruct'] is not None:
                structure_names.update(dicomutils.get_roi_names(database.dbabspath(imagedoc['rtstruct'])))

            # also get any custom masks (directly inserted as voxelized mask arrays)
            logger.info("Found structures: {}".format(structure_names))

            # generate missing masks
            for structure_name in structure_names:
                if structure_name in existing_structures:
                    continue
                logger.info('Requesting mask generation for structure "{}"'.format(structure_name))
                try:
                    existing_structures[structure_name] = database.structure_insert(imagedoc['_id'], name=structure_name)
                except:
                    logger.warning("Failed to create mask for structure \"{}\", most likely because it " \
                                   "doesn't contain any boundary coordinates.".format(structure_name))

            # add structure masking voxels of low density (air) within body contour
            air_struct_name = "T_AIR"
            if air_struct_name not in existing_structures:
                # force recompute
                ctvol, _ = database.get_ctvolume(imagedoc['_id'])
                air_mask = np.where(ctvol<0.2, 1, 0).astype(np.int8)
                existing_structures[air_struct_name] = database.structure_insert(imagedoc['_id'], air_struct_name, mask=air_mask)

            # refresh local copy of imagedoc and fetch mask data from it
            imagedoc = database.get_doc(DBCOLLECTIONS.IMAGES, imagedoc['_id'])

            # save to file
            try:
                logger.info('Saving masks for structures: {!s}'.format([s['name'] for s in imagedoc['structures']]))
                with h5py.File(self.args.out, 'w') as h5fd:
                    try:
                        for ii, structure in enumerate(tqdm(imagedoc['structures'], desc='Saving masks')):
                            sc = structure['boundbox']
                            f_mask = database.dbabspath(structure['maskfile'])
                            mask_arr = np.load(f_mask) # size matches CT
                            mask_name = structure['name']
                            # crop to the mask bbox for efficient storage
                            subslice = get_resizing_params(ic, sc)
                            cropped_mask = mask_arr[subslice]
                            assert list(cropped_mask.shape) == sc['size'][::-1]

                            group = h5fd.create_group(mask_name)
                            group.attrs['index'] = ii
                            group.attrs['name'] = mask_name
                            arrprops = group.create_group('ArrayProps')
                            arrprops.attrs['crop_size'] = cropped_mask.shape[::-1]
                            arrprops.attrs['crop_start'] = [sl.start for sl in subslice][::-1]
                            arrprops.attrs['size'] = ic['size']
                            group['mask'] = cropped_mask.astype(np.int8)

                            if self.args.numpy:
                                fname = pjoin(os.path.dirname(self.args.out), 'npmasks', mask_name+'.npy')
                                os.makedirs(os.path.dirname(fname), exist_ok=True)
                                np.save(fname, cropped_mask)
                    except:
                        logger.exception('Structure "{}" is missing its mask file'.format(structure['name']))
                        raise
            except:
                os.remove(self.args.out)

        @menucommand
        def fmaps(self, parser):
            parser.add_argument('--image_id', type=str, help="image database id")
            parser.add_argument('--image_uid', type=str, help="image dicom uid")
            parser.add_argument('--image_doi', type=str, help="image doi")
            parser.add_argument('--geom_id', type=str, help="geometry database id")
            parser.add_argument('--beamlist', type=str, help='file listing all beam ObjectIds to include')
            parser.add_argument('--out', '-o', type=str, default=os.path.curdir)
            self.args = parser.parse_args(namespace=self.args)

            from fmaps import Beam, Fmaps
            fmaps = Fmaps()

            beam_ids = self._get_beam_ids(beamlistfile=self.args.beamlist,
                                          image_id=self.args.image_id,
                                          image_uid=self.args.image_uid,
                                          image_doi=self.args.image_doi,
                                          geom_id=self.args.geom_id)

            for bb, beam_id in enumerate(tqdm(beam_ids, desc='Loading beams')):
                # download per-beamlet dose and construct sparse matrix as .mat file
                beamdoc = database.db[DBCOLLECTIONS.BEAMPHOTON].find_one({'_id': ObjectId(beam_id)})
                
                fluence_map = np.zeros([beamdoc['fmapdims'][1],beamdoc['fmapdims'][0]])
                for ii, beamletdoc in enumerate(beamdoc['beamlets']):
                    pos = beamletdoc['position']
                    fluence_map[pos[0], pos[1]] = 1.0

                fmaps.addBeam(Beam(fluence_map,
                                   gantry=beamdoc['angle_gantry'],
                                   couch=beamdoc['angle_couch'],
                                   coll=beamdoc['angle_coll'],
                              iso=[x/10 for x in beamdoc['isocenter']],
                              sad=beamdoc['sad']/10,
                              beamlet_size=[x/10 for x in beamdoc['beamletsize']],
                              beamlet_spacing=[x/10 for x in beamdoc['beamletspacing']],
                              ))

            fmaps.generate(self.args.out)

        @menucommand
        def exportdata(self, parser):
            parser.add_argument('--image_id', type=str, help="image database id")
            parser.add_argument('--image_uid', type=str, help="image dicom uid")
            parser.add_argument('--image_doi', type=str, help="image doi")
            parser.add_argument('--geom_id', type=str, help="geometry database id")
            parser.add_argument('--beamlist', type=str, help='file listing all beam ObjectIds to include')
            parser.add_argument('--nparticles', type=float, help='number of simulation histories')
            parser.add_argument('--tag', type=str, help='tag string referencing a set of simulations')
            parser.add_argument('--drop_thresh', default=None, type=none_or_type(float), help='drop dose values below this percent of each beamlet\'s max element')
            parser.add_argument('--magnetic_field', default=None, type=none_or_type(float), help='magnetic field strength in Z-direction (unit: Tesla)')
            parser.add_argument('--name', type=str, default='dose3d.bin', help='name of data file to export for every beamlet')
            parser.add_argument('--out', '-o', type=str, default=os.path.curdir)
            self.args = parser.parse_args()
            assert self.args.drop_thresh is None or self.args.drop_thresh >= 0.0

            data_filename = os.path.basename(self.args.name)

            # prepare data output
            if os.path.isdir(self.args.out):
                outfile = pjoin(self.args.out, os.path.splitext(data_filename)[0]+'.h5')
            else:
                outfile = self.args.out
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            assert os.path.isdir(os.path.dirname(outfile))

            imagedoc = database.get_image_doc(id=self.args.image_id,
                                              uid=self.args.image_uid,
                                              doi=self.args.image_doi)
            geomdoc = database.get_geometry_doc(id=self.args.geom_id,
                                                image_id=imagedoc['_id'])

            # calculate resizing params
            gc = geomdoc['coordsys']
            ic = imagedoc['coordsys']
            subslice = get_resizing_params(ic, gc)
            logger.debug('embedding data subarray with size {!s} into full array with size {!s} at {!s}'.format(
                gc['size'], ic['size'], '[{}]'.format(', '.join(['{}:{}'.format(sl.start,sl.stop) for sl in subslice][::-1]))))

            beam_ids = self._get_beam_ids(beamlistfile=self.args.beamlist,
                                          image_id=self.args.image_id,
                                          image_uid=self.args.image_uid,
                                          image_doi=self.args.image_doi,
                                          geom_id=self.args.geom_id)

            loadarr = ArrayLoader(multiproc=True, dosefactor=1.0, max_samples=1, data_filename=data_filename)

            # download per-beamlet dose and construct sparse matrix as .mat file
            sparsemat = SparseMatrixCOO(outfile, drop_thresh=self.args.drop_thresh)
            for bb, beam_id in enumerate(tqdm(beam_ids,  desc='Processing beams')):
                # supplementary information
                beamdoc = database.db[DBCOLLECTIONS.BEAMPHOTON].find_one({'_id': ObjectId(beam_id)})

                filters = []
                beamlet_ids = []
                for beamletdoc in sort_beamlets_columnorder(beamdoc):
                    beamlet_ids.append(beamletdoc['_id'])
                    filter = {'subbeam_id': ObjectId(beamletdoc['_id']),
                              'procstatus.status': {'$in': [PROCSTATUS.FINISHED, PROCSTATUS.SKIPPED]}}
                    if self.args.nparticles:
                        filter['num_particles'] = int(self.args.nparticles)
                    if self.args.magnetic_field:
                        filter['magnetic_field.2'] = self.args.magnetic_field,
                    if self.args.tag:
                        filter['tag'] = self.args.tag

                    filters.append(filter)

                simdocs = []
                for ifilter in filters:
                    simdoc = get_simdoc(ifilter)
                    simdocs.append(simdoc)

                if not all((doc is not None for doc in simdocs)):
                    nfinished = 0
                    for doc in simdocs:
                        if doc is not None: nfinished += 1
                    raise RuntimeError('Only {} of {} simulations for beam "{}" have been completed. ' \
                                       'Please wait for the rest to complete and try again.'.format(nfinished, len(simdocs), beam_id) )

                # check for requested data file
                nwithdata = 0
                nskipped = 0
                for simdoc in simdocs:
                    if simdoc['num_particles'] <= 0:
                        nskipped += 1
                    elif os.path.isfile(pjoin(database.build_datapath_sample(simdoc['_id'], simdoc['samples'][0]['_id']), data_filename)):
                        nwithdata += 1
                if (nwithdata + nskipped) < len(simdocs):
                    raise RuntimeError('{} of {} simulations for beam "{}" contain the requested data file: "{}".\n'
                                       'Try again with one of the following data filenames or wait and try later: {!s}'.format(
                        nwithdata + nskipped, len(simdocs), beam_id, data_filename,
                        os.listdir(database.build_datapath_sample(simdocs[0]['_id'], simdocs[0]['samples'][0]['_id']))
                    ))

                with multiprocessing.Pool() as pool:
                    def poolmap(chunksize):
                        def f(*args):
                            return pool.imap(*args, chunksize=chunksize)
                        return f
                    map_funcs = [poolmap(8), poolmap(1), map]
                    while len(map_funcs):
                        try:
                            map_func = map_funcs.pop(0)
                            iresults = map_func(loadarr, simdocs)

                            for ii, (result, simdoc) in enumerate(zip(tqdm(iresults, desc='Collecting beamlet data', total=len(simdocs)), simdocs)):
                                if result is None:
                                    raise RuntimeError('Error while loading beamlet data arrays for simulation "{!s}"'.format(simdoc['_id']))

                                # load noisy dose volume for one beamlet
                                dosearr = result[0][0][0]
                                if np.all(dosearr == 0):
                                    # add empty column
                                    sparsemat.add_column(None)
                                else:
                                    # resize to match CT coordsys
                                    fulldosearr = np.zeros(ic['size'][::-1])
                                    fulldosearr[subslice] = dosearr
                                    sparsemat.add_column(fulldosearr)
                            break

                        except multiprocessing.pool.MaybeEncodingError as err:
                            logger.warning('Data loading failed, falling back to less efficient method')
                            assert len(map_funcs)
                            continue

            logger.info('Writing sparse beamlet data matrix to file: "{}"'.format(outfile))
            sparsemat.finish()

        @menucommand
        def export_detection_data(self, parser):
            parser.add_argument('--image_id', type=str, help="image database id")
            parser.add_argument('--image_uid', type=str, help="image dicom uid")
            parser.add_argument('--image_doi', type=str, help="image doi")
            parser.add_argument('--geom_id', type=str, help="geometry database id")
            parser.add_argument('--beamlist', type=str, help='file listing all beam ObjectIds to include')
            parser.add_argument('--nparticles', type=float, help='number of simulation histories')
            parser.add_argument('--tag', type=str, help='tag string referencing a set of simulations')
            parser.add_argument('--name', type=str, default='detectedevents.pb', help='name of data file to aggregate from every beamlet')
            parser.add_argument('--out', '-o', type=str, default=os.path.curdir)
            self.args = parser.parse_args()

            # You need to copy your protobuf_pb2 file into ./dosecalc/webapi/ for this to work
            from protobuf_pb2 import pbDetectedEvents, pbDetectionEvent

            # strip any directory names from user-specified protobuf filename
            # This is the name of the file we will match against for each of the simulation outputs
            data_filename = os.path.basename(self.args.name)

            # prepare data output (either .mat or .h5 would be convenient for matlab users)
            if os.path.isdir(self.args.out):
                outfile = pjoin(self.args.out, os.path.splitext(data_filename)[0]+'.h5')
            else:
                outfile = self.args.out
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            assert os.path.isdir(os.path.dirname(outfile))

            # Begin our navigation of the mongodb database
            # database entries are called documents (docs). They act like python dictionaries after
            # we retrieve them with a .find() or .find_one() command (called inside my get_*_doc() functions)
            # for a complete list of the contents of each "doc" type, see the bottom of database.py
            # (gen_doc_*() functions define the database document structures)
            #
            # we essentially navigate the tree-like hierarchy of linked documents by repeatedly searching for the
            # child doc that has a matching id, stored in the parent doc
            # the tree looks like:  image -> geometry -> beam(s) -> beamlet(s) -> simulation(s) -> sample(s)
            imagedoc = database.get_image_doc(id=self.args.image_id,
                                              uid=self.args.image_uid,
                                              doi=self.args.image_doi)
            geomdoc = database.get_geometry_doc(id=self.args.geom_id,
                                                image_id=imagedoc['_id'])
            # this is a convenience function for matching to beam docs by one of the five options.
            # match by image_id/uid/doi is simplest (gives all beams for an image)
            # the _id is generated when the image is first inserted
            #     _uid is the Dicom uuid assigned when the image is captured by the scanner
            #     _doi is the "plan_name" you assign when running simpledose create-plan <doi> <data-dir>
            # match by beam_id is also possible for absolute control. --beamlist expects a text file with a
            # mongodb-assigned beam_id on each line. you can access the id of any document with doc['_id']
            beam_ids = self._get_beam_ids(beamlistfile=self.args.beamlist,
                                          image_id=self.args.image_id,
                                          image_uid=self.args.image_uid,
                                          image_doi=self.args.image_doi,
                                          geom_id=self.args.geom_id)

            # Let's pretend that you are simply appending the data from each DetectionEvent to the end of each
            # of the following four data arrays. You can probably find a way to write a list of structs to h5
            # just like you did for the protobuf file, but writing four equal-length (co-registered) arrays
            # is much simpler than writing a list of structs with h5.
            allGlobalTimes = []
            allEventIds    = []
            allDetectorIds = []
            allEnergies    = []
            allBeams       = []
            allBeamlets    = []

            # locate/read per-beamlet data and aggregate into final output file
            for bb, beam_id in enumerate(tqdm(beam_ids,  desc='Processing beams')):
                beamdoc = database.db[DBCOLLECTIONS.BEAMPHOTON].find_one({'_id': ObjectId(beam_id)})

                filters = []
                beamlet_ids = []
                # iterate over the beamlets in this beamdoc, sorted by the beamlet position in fmap
                for beamletdoc in sort_beamlets_columnorder(beamdoc):
                    beamlet_ids.append(beamletdoc['_id'])
                    # filters are used with .find() and .find_one() to access specific docs from the database
                    # these filters are saved to a list for now, and used in a multithreaded lookup function later
                    filter = {'subbeam_id': ObjectId(beamletdoc['_id']),
                              'procstatus.status': {'$in': [PROCSTATUS.FINISHED, PROCSTATUS.SKIPPED]}}
                    if self.args.nparticles:
                        filter['num_particles'] = int(self.args.nparticles)
                    if self.args.tag:
                        filter['tag'] = self.args.tag
                    filters.append(filter)

                # call get_simdoc() [at top of file] for each filter, using a multithreaded approach for speed
                #with multiprocessing.Pool() as pool:
                    #simdocs = pool.map(get_simdoc, filters) 
                #print(list(filters))              
                #simdocs = map(get_simdoc, filters)

                simdocs = []
                for ifilter in filters:
                    simdoc = get_simdoc(ifilter)
                    simdocs.append(simdoc)

                # pool.map is a blocking call, so we only get to this line after get_simdoc() has been called
                # for every filter. Now we look at the success/fail of each filter to make sure there are no
                # errors, otherwise we quit. Common error here is if some simulation tasks are still running.
                if not all((doc is not None for doc in simdocs)):
                    nfinished = 0
                    for doc in simdocs:
                        if doc is not None: nfinished += 1
                    raise RuntimeError('Only {} of {} simulations for beam "{}" have been completed. ' \
                                       'Please wait for the rest to complete and try again.'.format(nfinished, len(simdocs), beam_id) )

                # check for requested data file. We look in the per-simulation data folder in <dbdata> to confirm
                # that the requested data file exists for all sims. Error and exit otherwise.
                # common error here is a type in the filename, so some options are suggested to remind the user
                # of the actual files produced by geant4.
                nwithdata = 0
                nskipped = 0
                for simdoc in simdocs:
                    if simdoc['num_particles'] <= 0:
                        nskipped += 1
                    elif os.path.isfile(pjoin(database.build_datapath_sample(simdoc['_id'], simdoc['samples'][0]['_id']), data_filename)):
                        nwithdata += 1
                # if (nwithdata + nskipped) < len(simdocs):
                  #   raise RuntimeError('{} of {} simulations for beam "{}" contain the requested data file: "{}". Try again with one of the following data filenames: {!s}'.format(
                  #       nwithdata + nskipped, len(simdocs), beam_id, data_filename,
                  #       os.listdir(database.build_datapath_sample(simdocs[0]['_id'], simdocs[0]['samples'][0]['_id']))
                #  ))

                # now we know that all data files exist, we just need to read them to memory, aggregate somehow,
                # then write the data to the output file. I removed my complex multiprocessing approach to keep
                # this easy to understand. You can try to implement multiprocessing later if you want
                #print(simdocs)
                #print(list(simdocs))
                for bblets, simdoc in enumerate(simdocs):
                    if not simdoc['samples']:
                        continue
                    else:
                        # build the path leading to this sim's data file
                        sampledoc = simdoc['samples'][0]
                        sim_data_dir = database.build_datapath_sample(simdoc['_id'], sampledoc['_id'])
                        data_path = pjoin(sim_data_dir, data_filename)
                        # read the data from protobuf data file
                        with open(data_path, 'rb') as fd:
                            detected_events = pbDetectedEvents()
                            detected_events.ParseFromString(fd.read())
                            # merge this sim's data into an aggregate data structure
                            # (you should decide how to implement this. This is just an example)
                        # aggregate the data into memory
                        for event in detected_events.detectionEvent:
                            allGlobalTimes.append(event.globalTime)
                            allEventIds.append(event.eventId)
                            allDetectorIds.append(event.detectorId)
                            allEnergies.append(event.energy)
                            allBeams.append(bb)
                            allBeamlets.append(bblets)

            # heres a simple example of writing each of the four lists as a separate "dataset" in h5
            # This will make it easy to read all the data into matlab as four separate vectors
            with h5py.File(outfile, mode='w') as h5root:
                h5root.create_dataset('/globalTimes', data=allGlobalTimes)
                h5root.create_dataset('/eventIds', data=allEventIds)
                h5root.create_dataset('/detectorIds', data=allDetectorIds)
                h5root.create_dataset('/energy', data=allEnergies)
                h5root.create_dataset('/beamNo', data=allBeams)
                h5root.create_dataset('/beamletNo', data=allBeamlets)
            logger.info('Writing detected events to file: "{}"'.format(outfile))


        @menucommand
        def beamletdose_DL(self, parser):
            parser.add_argument('beamlist', type=str, help='file listing all beam ObjectIds to include')
            parser.add_argument('--nparticles', type=float, help='number of simulation histories')
            parser.add_argument('--drop_thresh', default=None, type=none_or_type(float), help='drop dose values below this percent of each beamlet\'s max element')
            parser.add_argument('--magnetic_field', default=1.5, type=float, help='magnetic field strength in Z-direction (unit: Tesla)')
            parser.add_argument('--out', '-o', type=str, default=os.path.curdir)
            parser.add_argument('--predict', nargs=3, help='--predict <config-file> <weights-file> <stats-file>')
            parser.add_argument('--nopredict', nargs=1, help='--nopredict <stats-file>')
            parser.add_argument('--zcontext', type=none_or_type(int), default=12, help='number of slices on each side of beamlet center (z-axis) to include in dosecalc')
            parser.add_argument('--xcontext', type=none_or_type(int), default=12, help='number of rows/cols on each side of beamlet center (x,y-axes) to include in dosecalc')
            parser.add_argument('--make-plots', action='store_true', default=False, help='save debug plots indicating prediction output')
            parser.add_argument('--cpu', action='store_true', help='inference on CPU')
            self.args = parser.parse_args(namespace=self.args)
            assert self.args.predict or self.args.nopredict
            assert self.args.drop_thresh is None or self.args.drop_thresh >= 0.0
            assert self.args.zcontext is None or self.args.zcontext > 0
            assert self.args.xcontext is None or self.args.xcontext > 0
            make_plots = self.args.make_plots
            #  make_plots = True

            # prepare data output
            if os.path.isdir(self.args.out):
                if self.args.predict:
                    outfile = pjoin(self.args.out, 'beamlet_dose_predicted.h5')
                else:
                    outfile = pjoin(self.args.out, 'beamlet_dose.h5')
            else:
                outfile = self.args.out
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            assert os.path.isdir(os.path.dirname(outfile))

            # prepare dose prediction model
            model = None
            if self.args.predict:
                if self.args.cpu:
                    os.environ['CUDA_VISIBLE_DEVICES'] = ""
                config_file, weights_file, stats_file = self.args.predict
                # docker volume mounting magic places the mcdose module in this src directory
                from mcdose import get_trained_model
                mcdoselogger = logging.getLogger('MCDose')
                mcdoselogger.addHandler(logging.StreamHandler())
                mcdoselogger.setLevel(logging.DEBUG)
                model = get_trained_model(
                    config=config_file,
                    weights=weights_file,
                    normstats=stats_file,
                )
            else:
                stats_file = self.args.nopredict[0]

            with open(stats_file, 'r') as fd:
                normstats = json.load(fd)
                if 'factor' not in normstats:
                    normstats['factor'] = [1.0]*len(normstats['mean'])

            with open(self.args.beamlist, 'r') as fd:
                beam_ids = [line.rstrip('\n') for line in fd]

            loadarr = ArrayLoader(reorient=True, context=(self.args.xcontext, 60, self.args.zcontext),
                                  get_geom=True, get_label=make_plots, reversible=True, multiproc=True,
                                  dosefactor=normstats['factor'][0])

            # download per-beamlet dose and construct sparse matrix as .mat file
            sparsemat = SparseMatrixCOO(outfile, drop_thresh=self.args.drop_thresh)
            for bb, beam_id in enumerate(tqdm(beam_ids,  desc='Processing beams')):
                # supplementary information
                beamdoc = database.db[DBCOLLECTIONS.BEAMPHOTON].find_one({'_id': ObjectId(beam_id)})
                geomdoc = database.db[DBCOLLECTIONS.MCGEOM].find_one({'_id': ObjectId(beamdoc['geom_id'])})
                arrsize = geomdoc['coordsys']['size']
                theta = beamdoc['angle_gantry']

                filters = []
                beamlet_ids = []
                for beamletdoc in sort_beamlets_columnorder(beamdoc):
                    beamlet_ids.append(beamletdoc['_id'])
                    filter = {'subbeam_id': ObjectId(beamletdoc['_id']), 'magnetic_field.2': self.args.magnetic_field}
                    if self.args.nparticles:
                        filter['num_particles'] = int(self.args.nparticles)
                    filters.append(filter)

                with multiprocessing.Pool() as pool:
                    simdocs = pool.map(get_simdoc, filters)

                with multiprocessing.Pool() as pool:
                    iresults = pool.imap(loadarr, simdocs, chunksize=8)
                    #  iresults = (loadarr(simdoc) for simdoc in simdocs)
                    for ii, (result, simdoc) in enumerate(zip(tqdm(iresults, desc='Collecting dose data', total=len(simdocs)), simdocs)):
                        if result is None:
                            raise RuntimeError('Error while loading beamlet dose arrays for simulation "{!s}"'.format(simdoc['_id']))

                        if model is None:
                            # load noisy dose volume for one beamlet
                            rotdosearr, unprocessor = result[0][0][0], result[1]
                            # force all dose to go through rotations to match aliasing effects (for now)
                            dosearr = unprocessor(rotdosearr)
                            dosearr[dosearr<0.0] = 0.0 # enforce realistic dose (aliasing and prediction may contain neg. artifacts)

                        else:
                            # predict clean dose from noisy dose
                            arrs, unprocessor = result
                            dosearr, geomarr = arrs[0][0], arrs[1]

                            # predict, rotate back (already rotated by loadarr)
                            inputs  = (np.stack((dosearr, geomarr), axis=-1)[None,...]).astype(np.float32)
                            rotpredarr = model(inputs).numpy()[0,...,0]
                            predarr = unprocessor(rotpredarr) # drop "example" and "channel" axes
                            predarr[predarr<0.0] = 0.0 # enforce realistic dose (aliasing and prediction may contain neg. artifacts)

                            if make_plots and ii < 10:
                                gtrutharr = arrs[2] # drop channel axis

                                import matplotlib.pyplot as plt
                                from mcdose.visualize import create_volume_dose_figure, save_figure_array
                                figimg = create_volume_dose_figure(
                                    np.stack([
                                        np.stack([
                                            gtrutharr[sliceidx],
                                            dosearr  [sliceidx],
                                            predarr  [sliceidx],
                                            geomarr  [sliceidx],
                                            predarr[sliceidx] - gtrutharr[sliceidx],
                                        ], axis=0) for sliceidx in range(inputs.shape[1])
                                    ], axis=0),
                                    dpi=200,
                                    col_labels=['ground truth', 'input', 'predict', 'geom', 'predict - input'],
                                    own_scale=False
                                )
                                fig_outdir = os.path.splitext(outfile)[0]+'_figs'
                                os.makedirs(fig_outdir, exist_ok=True)
                                save_figure_array(figimg, pjoin(fig_outdir, 'beam{:04d}_blt{:05d}.png'.format(bb, ii)))

                            # replace noisy dose with predicted dose in sparse matrix
                            dosearr = predarr

                        sparsemat.add_column(dosearr)

            logger.info('Writing sparse beamlet dose matrix to file: "{}"'.format(outfile))
            sparsemat.finish()

        @menucommand
        def finaldose(self, parser):
            """sum the beamlet-dose volumes for the specified beams and save to file"""
            parser.add_argument('--image_id', type=str, help="image database id")
            parser.add_argument('--image_uid', type=str, help="image dicom uid")
            parser.add_argument('--image_doi', type=str, help="image doi")
            parser.add_argument('--geom_id', type=str, help="geometry database id")
            parser.add_argument('--beamlist', type=str, help='file listing all beam ObjectIds to include')
            parser.add_argument('--nparticles', type=float, help='number of simulation histories')
            parser.add_argument('--magnetic_field', default=None, type=none_or_type(float), help='magnetic field strength in Z-direction (unit: Tesla)')
            parser.add_argument('--out', '-o', type=str, default=os.path.curdir)
            self.args = parser.parse_args(namespace=self.args)
            outdir = self.args.out

            imagedoc = database.get_image_doc(id=self.args.image_id,
                                              uid=self.args.image_uid,
                                              doi=self.args.image_doi)
            geomdoc = database.get_geometry_doc(id=self.args.geom_id,
                                                image_id=imagedoc['_id'])

            beam_ids = self._get_beam_ids(beamlistfile=self.args.beamlist,
                                          image_id=self.args.image_id,
                                          image_uid=self.args.image_uid,
                                          image_doi=self.args.image_doi,
                                          geom_id=self.args.geom_id)

            # calculate resizing params
            gc = geomdoc['coordsys']
            ic = imagedoc['coordsys']
            subslice = get_resizing_params(ic, gc)
            logger.debug('embedding dose subarray with size {!s} into full array with size {!s} at {!s}'.format(
                gc['size'], ic['size'], '[{}]'.format(', '.join(['{}:{}'.format(sl.start,sl.stop) for sl in subslice][::-1]))))

            # get simulation samples one by one
            subarrsize = gc['size']
            sumarrs = {('finaldose', 'dosefile'): np.zeros(subarrsize[::-1]),
                       ('density',   'densfile'): np.zeros(subarrsize[::-1]), }

            for bb, beam_id in enumerate(tqdm(beam_ids, desc='Processing beams')):
                # download per-beamlet dose and sum together
                beamdoc = database.get_doc(DBCOLLECTIONS.BEAMPHOTON, beam_id)
                assert ObjectId(beamdoc['geom_id']) == ObjectId(geomdoc['_id'])

                nbeamletdocs = len(beamdoc['beamlets'])
                for ii, beamletdoc in enumerate(tqdm(beamdoc['beamlets'], desc='Summing beamlets')):
                    filter = {
                        'subbeam_id': ObjectId(beamletdoc['_id']),
                    }
                    if self.args.nparticles:
                        filter['num_particles'] = int(self.args.nparticles)
                    if self.args.magnetic_field:
                        filter['magnetic_field.2'] = self.args.magnetic_field,
                    simdoc = database.db[DBCOLLECTIONS.SIMULATION].find_one(filter)
                    for key in sumarrs.keys():
                        arrlabel, arrtype = key
                        if arrtype == 'densfile' and ii>0:
                            continue
                        datafile = database.dbabspath(simdoc['samples'][0][arrtype])

                        with open(datafile, 'rb') as fd:
                            buf = fd.read()
                        dtype = 'f4' if arrtype == 'densfile' else 'f8'
                        arr = np.frombuffer(buf, dtype).reshape(subarrsize[::-1])
                        sumarrs[key] += arr

            for key, sumarr in sumarrs.items():
                arrlabel, arrtype = key
                fullarr = np.zeros(ic['size'][::-1])
                fullarr[subslice] = sumarr
                sumarrs[key] = fullarr

            os.makedirs(outdir, exist_ok=True)
            for (arrlabel, arrtype), sumarr in sumarrs.items():
                np.save(pjoin(outdir, '{}.npy'.format(arrlabel)), sumarr)
                save_bin(pjoin(outdir, '{}.raw'.format(arrlabel)), sumarr)

        #TODO convert rest to db api
        @menucommand
        def quan_mcconfig(self, parser):
            raise NotImplementedError("need to convert from rest api to db api")
            parser.add_argument('beam_id', help='Beam ObjectId, or file with beam id on each line')
            parser.add_argument('--out', '-o', type=str, default=os.path.curdir)
            self.args = parser.parse_args(namespace=self.args)

            beam_ids = []
            if os.path.isfile(self.args.beam_id):
                with open(self.args.beam_id, 'r') as fd:
                    for beam_id in fd:
                        beam_ids.append(beam_id.rstrip())
            else:
                beam_ids.append(self.args.beam_id)

            for beam_id in beam_ids:
                outdir = pjoin(self.args.out, beam_id)
                os.makedirs(outdir, exist_ok=True)

                # download per-beamlet dose and sum together
                p = payloadtypes.RESTReqBeamPhotonGet()
                p.beam_id = beam_id
                p.recursive = False
                beamdoc = validate_rest_response(p.send_request(dsaddr))

                beamlets = [blt['position'] for blt in beamdoc['beamlets']]
                beamlets.sort(key=lambda p: p[1]*1000+p[0])

                # define beams and control points
                cp = quan_config.ControlPoint()
                cp.gantry_rot = beamdoc['angle_gantry']
                cp.mu = 1.0 # replace with 1/#cp
                cp.sad = beamdoc['sad']
                cp.iso = beamdoc['isocenter']
                #  cp.xjaw_pos = tuple([d*(beamdoc['fmapdims'][0]/2)*beamdoc['beamletsize'][0] for d in [-1.0, 1.0]])
                #  cp.yjaw_pos = tuple([d*(beamdoc['fmapdims'][1]/2)*beamdoc['beamletsize'][1] for d in [-1.0, 1.0]])
                # split into multiple control points to fill holes in target projection
                leaf_edge_seqs = quan_config.get_leaf_edges(beamlets, beamdoc['fmapdims'], beamdoc['beamletsize'])
                cps = []
                for id, seq in enumerate(leaf_edge_seqs):
                    cp_copy = copy.deepcopy(cp)
                    cp_copy.id = id+1
                    cp_copy.leaf_edges = seq
                    cp_copy.xjaw_pos, cp_copy.yjaw_pos = quan_config.get_jaw_positions(seq, beamdoc['fmapdims'], beamdoc['beamletsize'])
                    cp_copy.mu = 1/len(leaf_edge_seqs)
                    cps.append(cp_copy)
                mlcbeams = [quan_config.MLCBeam()]
                mlcbeams[-1].control_points = cps

                quan_config.generate_mlcdef(pjoin(outdir, 'mlcdef_{!s}.txt'.format(beam_id)), beamdoc['fmapdims'], beamdoc['beamletsize'])
                quan_config.generate_rtplan(outdir, mlcbeams, fsuffix=beam_id)

    #====================================
    # SINGLE-USE FUNCTIONS
    #====================================
    @menucommand
    def userfunc(self, parser):
        """>> Collection of single-use user functions for brute forcing database change"""
        self.CMenuUserFunctions()

    class CMenuUserFunctions(CommandMenuBase):
        @menucommand
        def update_nruns(self, parser):
            db = database.db
            beamdocs = db[DBCOLLECTIONS.BEAMPHOTON].find({'mlrole': 'test'})

            for beam in beamdocs:
                for beamlet in beam['beamlets']:
                    for simulation in beamlet['simulations']:
                        result = db[DBCOLLECTIONS.SIMULATION].update_one({'_id': simulation}, update={'$set': {'num_runs': 1}})
                        if not result.matched_count:
                            raise RuntimeError('Failed to modify simulation "{!s}"'.format(simulation))

        @menucommand
        def add_sim_tasks(self, parser):
            """Add new sims to all beams matching MLROLE"""
            parser.add_argument('--role', required=True, choices=[MLROLE.TRAIN, MLROLE.TEST], type=str, help='Filter for beams to which to add sims')
            parser.add_argument('--nparticles', required=True, type=int, nargs="+", help='Number of particles to simulate (may supply multiple)')
            parser.add_argument('--vartype', choices=[VARTYPE.HIGH, VARTYPE.LOW], default='highvar', help='type of sample')
            parser.add_argument('--nsamples', type=int, default=1, help='number of simulations to run')
            parser.parse_args(namespace=self.args)

            beamids = database.db[DBCOLLECTIONS.BEAMPHOTON].aggregate([
                {'$match': {'mlrole': self.args.role}},
                {'$project': {'_id': True}},
                {'$group': {'_id': None, 'ids': {'$addToSet': '$_id'}}}
            ]).next()['ids']
            print('Adding simulation tasks for {} beams'.format(len(beamids)))
            for nparticles in self.args.nparticles:
                print('Adding for {} histories'.format(nparticles))
                for beamid in beamids:
                    database.add_sims_to_beam(beam_id=beamid, vartype=self.args.vartype, num_runs=self.args.nsamples,
                                              num_particles=nparticles)
        @menucommand
        def test_code(self, parser):
            image_id = "5d9bd5ec4f71a0917827897e"
            geom_id = "5d9bd5f04f71a09178278980"
            out = "/media/hdd1/g4sim/beamlet_dose/HN010/1.5T/10e6/test"

            # check volumes
            ctvolume, _ = database.get_ctvolume(image_id=image_id)
            np.save(pjoin(out, "ctvolume.npy"), ctvolume)
            geomvolume, _ = database.get_geometry_volume(geom_id=geom_id)
            np.save(pjoin(out, "geomvolume.npy"), geomvolume)

            # check mcgeom
            import geometry
            mcgeom_out = pjoin(out, "mcgeom.txt")
            mcgeomvol_out = pjoin(out, "mcgeomvolume.npy")
            if not os.path.exists(mcgeom_out):
                geometry.generate_geometry(mcgeom_out, geomvolume, (2.5, 2.5, 2.5))
            if not os.path.exists(mcgeomvol_out):
                with open(mcgeom_out) as fd:
                    size = [int(x) for x in fd.readline().strip('\n').split(' ')]
                    voxelsize = [float(x) for x in fd.readline().strip('\n').split(' ')]
                    start = [float(x) for x in fd.readline().strip('\n').split(' ')]
                    mcgeomvolume = np.zeros(size[::-1]).ravel()
                    for ii, line in enumerate(fd):
                        mcgeomvolume[ii] = float(line.strip('\n').split(' ')[0])
                    mcgeomvolume = mcgeomvolume.reshape(size[::-1])
                    np.save(mcgeomvol_out, mcgeomvolume)

        @menucommand
        def diff_beamlets(self, parser):
            parser.add_argument('beamlists', nargs=2, type=str)
            self.args =parser.parse_args(namespace=self.args)

            beamlists = []
            for beamlist_file in self.args.beamlists:
                with open(beamlist_file, 'r') as fd:
                    beamlist=[]
                    for line in fd:
                        beamlist.append(ObjectId(line.strip('\n')))
                    beamlists.append(beamlist)

            for beama, beamb in zip(*beamlists):
                beamdoca = database.db[DBCOLLECTIONS.BEAMPHOTON].find_one({'_id': beama})
                beamdocb = database.db[DBCOLLECTIONS.BEAMPHOTON].find_one({'_id': beamb})
                assert beamdoca and beamdocb

                # extract beamlets
                def extract_beamlets(beamdoc):
                    beamlets = []
                    for blt in beamdoc['beamlets']:
                        beamlets.append(blt['position'])
                    return beamlets

                beamletsa = extract_beamlets(beamdoca)
                beamletsb = extract_beamlets(beamdocb)

                assert database.diff_fluence_arrays(beamletsa, beamletsb, verbose=True)

        @menucommand
        def test_new_raytrace(self, parser):
            # get ctdoc, geomdoc, structure_id to test
            imdoc = database.db[DBCOLLECTIONS.IMAGES].find_one({'doi': "HN010"})
            geomdoc = database.db[DBCOLLECTIONS.MCGEOM].find_one({'image_id': imdoc['_id']})

            beamlists = {
                'P_PTV_5400': [ObjectId(x) for x in [
                    '5e165b29f7aee215815be409',
                    '5e165b29f7aee215815be532',
                    '5e165b2af7aee215815be641',
                    '5e165b2af7aee215815be7bd',
                    '5e165b2af7aee215815be93e',
                    '5e165b2af7aee215815bea38',
                    '5e165b2af7aee215815beb80',
                ]],
                'P_PTV_5940': [ObjectId(x) for x in [
                    '5e165c05f7aee215815c088c',
                    '5e165c05f7aee215815c0a96',
                    '5e165c06f7aee215815c0d2d',
                    '5e165c06f7aee215815c0f3c',
                    '5e165c06f7aee215815c10ea',
                    '5e165c07f7aee215815c134d',
                    '5e165c07f7aee215815c15ba',
                ]],
            }

            for ptvname, beamlist in beamlists.items():
                beamdocs = list(database.db[DBCOLLECTIONS.BEAMPHOTON].find({'_id': {"$in": beamlist}}))
                structdoc = None
                for struct in imdoc['structures']:
                    if struct['name'] == ptvname:
                        structdoc = struct

                assert len(beamdocs)
                for x in (imdoc, geomdoc, structdoc, beamdocs[0]):
                    assert '_id' in x

                # calcluate new raytrace result
                import geometry
                for bb, beamdoc in enumerate(beamdocs):
                    mask = database.get_structure_mask(imdoc['_id'], structdoc['_id']) # full sized mask
                    active_beamlets, fmap = geometry.get_active_beamlets(
                        mask=mask,
                        angle_gantry=beamdoc['angle_gantry'],
                        angle_couch=beamdoc['angle_couch'],
                        angle_coll=beamdoc['angle_coll'],
                        iso=beamdoc['isocenter'],
                        start=imdoc['coordsys']['start'],
                        spacing=imdoc['coordsys']['spacing'],
                        fmapdims=beamdoc['fmapdims'],
                        beamletspacing=beamdoc['beamletspacing'],
                        beamletsize=beamdoc['beamletsize'],
                        sad=beamdoc['sad'],
                    )

                    # extract old raytrace result
                    old_active_beamlets = []
                    for blt in beamdoc['beamlets']:
                        old_active_beamlets.append(blt['position'])

                    database.diff_fluence_arrays(active_beamlets, old_active_beamlets)

                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    ax = fig.subplots(2,2)
                    fmapnew = database.make_fluence_array(active_beamlets, beamdoc['fmapdims'])
                    ax[0,0].imshow(fmapnew)
                    ax[0,1].imshow(fmap)
                    fmapold = database.make_fluence_array(old_active_beamlets, beamdoc['fmapdims'])
                    ax[1,0].imshow(fmapold)
                    im =ax[1,1].imshow(fmapnew-fmapold, cmap='RdBu')
                    plt.colorbar(im)
                    fig.savefig('test_{}_{}.png'.format(ptvname, bb))

        @menucommand
        def patchbeamlets(self, parser):
            """resolve differences between existing beams and replacement fluence maps by deleting unnecessary
            beamlets and adding newly introduced beamlets
            """
            #  for struct in database.db[DBCOLLECTIONS.IMAGES].find_one({"_id": ObjectId("5d9bd5ec4f71a0917827897e")})['structures']:
            #      print('{:30s} {!s:50s}'.format(struct['name'], struct['_id']))
            #  beamdocs = database.db[DBCOLLECTIONS.BEAMPHOTON].find({"geom_id": ObjectId("5e16564100ad46279500ed4b")})
            #  for beam in beamdocs:
            #      print(beam['_id'], len(beam['beamlets']))
            #  return

            beams = [
                {
                    "fmaps_folder": "/media/hdd1/dosecalc_debug/ryan_fmaps/HN010/fluence_maps_5400/",
                    'magnetic_field': [0,0,0.0,'tesla'],
                     'beam_ids': {
                         "5e165b5df7aee215815bf62e": "fmap-000000.raw",
                         "5e165b5df7aee215815bf757": "fmap-000001.raw",
                         "5e165b5ef7aee215815bf866": "fmap-000002.raw",
                         "5e165b5ef7aee215815bf9e2": "fmap-000003.raw",
                         "5e165b5ef7aee215815bfb63": "fmap-000004.raw",
                         "5e165b5ff7aee215815bfc5d": "fmap-000005.raw",
                         "5e165b5ff7aee215815bfda5": "fmap-000006.raw",
                     }
                }, {
                    "fmaps_folder": "/media/hdd1/dosecalc_debug/ryan_fmaps/HN010/fluence_maps_5400/",
                    'magnetic_field': [0,0,1.5,'tesla'],
                     'beam_ids': {
                         "5e165b29f7aee215815be409": "fmap-000000.raw",
                         "5e165b29f7aee215815be532": "fmap-000001.raw",
                         "5e165b2af7aee215815be641": "fmap-000002.raw",
                         "5e165b2af7aee215815be7bd": "fmap-000003.raw",
                         "5e165b2af7aee215815be93e": "fmap-000004.raw",
                         "5e165b2af7aee215815bea38": "fmap-000005.raw",
                         "5e165b2af7aee215815beb80": "fmap-000006.raw",
                     }
                }, {
                    "fmaps_folder": "/media/hdd1/dosecalc_debug/ryan_fmaps/HN010/fluence_maps_5940/",
                    'magnetic_field': [0,0,0.0,'tesla'],
                     'beam_ids': {
                         "5e165c24f7aee215815c2639": "fmap-000000.raw",
                         "5e165c25f7aee215815c2843": "fmap-000001.raw",
                         "5e165c25f7aee215815c2ada": "fmap-000002.raw",
                         "5e165c25f7aee215815c2ce9": "fmap-000003.raw",
                         "5e165c25f7aee215815c2e97": "fmap-000004.raw",
                         "5e165c26f7aee215815c30fa": "fmap-000005.raw",
                         "5e165c26f7aee215815c3367": "fmap-000006.raw",
                     }
                }, {
                    "fmaps_folder": "/media/hdd1/dosecalc_debug/ryan_fmaps/HN010/fluence_maps_5940/",
                    'magnetic_field': [0,0,1.5,'tesla'],
                     'beam_ids': {
                         "5e165c05f7aee215815c088c": "fmap-000000.raw",
                         "5e165c05f7aee215815c0a96": "fmap-000001.raw",
                         "5e165c06f7aee215815c0d2d": "fmap-000002.raw",
                         "5e165c06f7aee215815c0f3c": "fmap-000003.raw",
                         "5e165c06f7aee215815c10ea": "fmap-000004.raw",
                         "5e165c07f7aee215815c134d": "fmap-000005.raw",
                         "5e165c07f7aee215815c15ba": "fmap-000006.raw",
                     }
                }
            ]

            for beam in beams:
                for bb, (beam_id, fmapfile) in enumerate( beam['beam_ids'].items() ):
                    fmaps_folder = beam['fmaps_folder']
                    fmapfile = pjoin(fmaps_folder, fmapfile)
                    fmap = np.squeeze(load_bin(fmapfile, (1, 40, 40)))
                    beamdoc = database.db[DBCOLLECTIONS.BEAMPHOTON].find_one({'_id': ObjectId(beam_id)})
                    print(beam_id, len(beamdoc['beamlets']),  np.count_nonzero(fmap))

                    # create fmap from database
                    fmap_db = np.zeros((40,40))
                    for blt in beamdoc['beamlets']:
                        y, x = blt['position']
                        fmap_db[y, x] = 1.0

                    save_bin(pjoin(fmaps_folder, 'fmap_from_db-{:06d}.raw'.format(bb)), fmap_db)

                    # delete unnecessary beamlets
                    for blt in beamdoc['beamlets']:
                        if fmap[y, x] <= 0.0:
                            print('deleting beamlet: [x={}, y={}]'.format(x, y))
                            database.subbeam_delete(beam_id=beam_id, subbeam_id=blt['_id'])

                    # add new beamlets
                    positions_yx = []
                    for y in range(fmap.shape[0]):
                        for x in range(fmap.shape[1]):
                            if fmap[y, x] > 0 and fmap_db[y, x] <= 0:
                                print('adding beamlet: [x={}, y={}]'.format(x, y))
                                positions_yx.append((y, x))

                    # populate new beamlets with simulation specs
                    subbeam_ids = database.subbeam_insert(beam_id=beam_id, positions=positions_yx)
                    for subbeam_id in subbeam_ids:
                        sim_id = database.simulation_insert(beam_id=beam_id,
                                                            subbeam_id=subbeam_id,
                                                            vartype=VARTYPE.LOW,
                                                            num_particles=1e5,
                                                            magnetic_field=beam['magnetic_field'],
                                                            num_runs=1)

        @menucommand
        def insert_sims_by_geometry(self, parser):
            """Insert a set of simulation requests for all beamlets assigned to a geometry"""
            parser.add_argument("geom_id", type=str)
            self.args = parser.parse_args(namespace=self.args)

            geomdoc = database.get_doc(DBCOLLECTIONS.MCGEOM, self.args.geom_id)
            if geomdoc is None:
                raise RuntimeError("Couldn't find geometry \"{}\"".format(self.args.geom_id))

            #  simdocs = database.db[DBCOLLECTIONS.SIMULATION].find({
            #      '$and': [
            #          {'geom_id': ObjectId(geomdoc['_id'])},
            #          {'num_particles': {'$lt': 20000}},
            #          {'date_added': {'$gte': datetime.today()-timedelta(hours=1)}},
            #      ]
            #  })
            #  for simdoc in simdocs:
            #      database.simulation_delete(simdoc['_id'])

            beams = [
                {
                    "fmaps_folder": "/media/hdd1/dosecalc_debug/ryan_fmaps/HN011/fluence_maps/",
                    'magnetic_field': [0,0,1.5,'tesla'],
                     'beam_ids': {
                         "5e15407dc9c073745a01cb7c": "fmap-000000.raw",
                         "5e15407dc9c073745a01cc75": "fmap-000001.raw",
                         "5e15407ec9c073745a01cd97": "fmap-000002.raw",
                         "5e15407ec9c073745a01ced3": "fmap-000003.raw",
                         "5e15407fc9c073745a01cfe6": "fmap-000004.raw",
                         "5e15407fc9c073745a01d0f4": "fmap-000005.raw",
                         "5e154080c9c073745a01d229": "fmap-000006.raw",
                     }
                }, {
                    "fmaps_folder": "/media/hdd1/dosecalc_debug/ryan_fmaps/HN010/fluence_maps_5400/",
                    'magnetic_field': [0,0,1.5,'tesla'],
                     'beam_ids': {
                         "5e165b29f7aee215815be409": "fmap-000000.raw",
                         "5e165b29f7aee215815be532": "fmap-000001.raw",
                         "5e165b2af7aee215815be641": "fmap-000002.raw",
                         "5e165b2af7aee215815be7bd": "fmap-000003.raw",
                         "5e165b2af7aee215815be93e": "fmap-000004.raw",
                         "5e165b2af7aee215815bea38": "fmap-000005.raw",
                         "5e165b2af7aee215815beb80": "fmap-000006.raw",
                     }
                }, {
                    "fmaps_folder": "/media/hdd1/dosecalc_debug/ryan_fmaps/HN010/fluence_maps_5940/",
                    'magnetic_field': [0,0,1.5,'tesla'],
                     'beam_ids': {
                         "5e165c05f7aee215815c088c": "fmap-000000.raw",
                         "5e165c05f7aee215815c0a96": "fmap-000001.raw",
                         "5e165c06f7aee215815c0d2d": "fmap-000002.raw",
                         "5e165c06f7aee215815c0f3c": "fmap-000003.raw",
                         "5e165c06f7aee215815c10ea": "fmap-000004.raw",
                         "5e165c07f7aee215815c134d": "fmap-000005.raw",
                         "5e165c07f7aee215815c15ba": "fmap-000006.raw",
                     }
                }
            ]
            for beam in beams:
                for beam_id in beam['beam_ids'].keys():
                    #  beamdocs = database.db[DBCOLLECTIONS.BEAMPHOTON].find({'geom_id': ObjectId(self.args.geom_id)})
                    beamdoc = database.get_doc(DBCOLLECTIONS.BEAMPHOTON, beam_id)
                    for subbeam in beamdoc['beamlets']:
                        for num_particles in [500, 1000, 2000, 5000]:
                            database.simulation_insert(beam_id=beamdoc['_id'],
                                                       subbeam_id=subbeam['_id'],
                                                       vartype=VARTYPE.LOW if num_particles>=1e5 else VARTYPE.HIGH,
                                                       num_particles=num_particles,
                                                       magnetic_field=[0,0,1.5,'tesla'],
                                                       num_runs=1)

        @menucommand
        def relocate_density_files(self, parser):
            """iterate through all beamlets and relocate density file to mcgeometry folder where it can be shared
            to save on disk space """
            for geomdoc in database.db[DBCOLLECTIONS.MCGEOM].find():
                geom_id = geomdoc['_id']

                densfile_path = pjoin(database.build_datapath_geom(geom_id), 'InputDensity.bin')
                print("Shared density path is \"{}\"".format(densfile_path))
                densfile_exists = os.path.isfile(densfile_path)
                for simdoc in database.db[DBCOLLECTIONS.SIMULATION].find({'geom_id': geom_id}):
                    for sampledoc in simdoc['samples']:
                        oldfile = database.dbabspath(sampledoc['densfile'])
                        if not densfile_exists:
                            # copy this density file to shared location
                            try: shutil.copy2(oldfile, densfile_path)
                            except Exception as e: print(e)
                            densfile_exists = True
                            print('copied file from "{} to "{}'.format(oldfile, densfile_path))

                        # redirect path to shared file (convert to relative path if necessary)
                        newfile_rel = database.dbrelpath(densfile_path)
                        if sampledoc['densfile'] != newfile_rel:
                            res = database.db[DBCOLLECTIONS.SIMULATION].update_many(
                                filter={
                                    '_id': simdoc['_id'],
                                    'samples._id': sampledoc['_id'],
                                },
                                update={
                                    '$set': {'samples.$.densfile': newfile_rel}
                                }
                            )
                            print('updated sample "{}" densfile path to "{}"'.format(sampledoc['_id'], newfile_rel))

                        # delete redundant file (don't delete relocated/shared file)
                        if oldfile != densfile_path:
                            try: os.remove(oldfile)
                            except: print('failed to delete file: \"{}\"'.format(oldfile))
                            print('deleted redundant densfile "{}"'.format(oldfile))

                        # delete additional large/wasteful files
                        for fname in ('run_log.txt', ):
                            filepath = pjoin(database.build_datapath_simulation(simdoc['_id']), fname)
                            try: os.remove(filepath)
                            except Exception as e:
                                print('failed to delete file: \"{}\"'.format(filepath))
                            print('deleted unnecessary file "{}"'.format(filepath))

        # TODO
        @menucommand
        def remove_duplicate_beams(self, parser):
            raise NotImplementedError()
            repmap = namedtuple("repmap", ("old", "new"))
            replace_map = [
                repmap("5dd5c9964b45cc99f1aa8252", "5dd5ca464b45cc99f1aab169"),
                repmap("5dd5c9974b45cc99f1aa834b", "5dd5ca474b45cc99f1aab262"),
                repmap("5dd5c9984b45cc99f1aa846d", "5dd5ca484b45cc99f1aab384"),
                repmap("5dd5c9984b45cc99f1aa85a9", "5dd5ca484b45cc99f1aab4c0"),
                repmap("5dd5c9984b45cc99f1aa86bc", "5dd5ca494b45cc99f1aab5d3"),
                repmap("5dd5c9984b45cc99f1aa87ca", "5dd5ca4a4b45cc99f1aab6e1"),
                repmap("5dd5c9994b45cc99f1aa88ff", "5dd5ca4a4b45cc99f1aab816"),
            ]
            for rep in replace_map:
                # get old beamdoc
                # for each beamlet in oldbeamdoc
                    # match to correct beamlet in newbeamdoc (by position)
                    # insert simulation objects into newbeamdoc array, and change beam_id and subbeam_id for these simulations
                    # move simulation data from oldbeamdoc to newbeamdoc folder (optional)
                # delete old beams docs
                pass

        @menucommand
        def insert_new_training_data(self, parser):
            """Added on 12jan2020 to create new geometry for all images/geoms designated for training/validation
            Since original geometry had issues with flipped z-axis and swapped beamlet position x and y indices"""
            # for all HN data associated with train/validate, add new geometry objects and re-generate random assortment
            # of beams, beamlets, and simulations associated with those new geometries
            image_uids = []
            for imdoc in database.db[DBCOLLECTIONS.IMAGES].find({'doi': {'$not': {"$in": ['HN010', 'HN011']}}}):
                print(imdoc['doi'], imdoc['_id'])
                image_uids.append(imdoc['_id'])

            print()
            gfd = open('corrupt_geoms.txt', 'w')
            for image_id in image_uids:
                print("Image: {}".format(image_id))
                # cleanup half-completed geoms
                for geomdoc in database.db[DBCOLLECTIONS.MCGEOM].find({
                        'image_id': image_id,
                        'date_added': {'$gte': datetime.today()-timedelta(hours=2)},
                }):
                    print('deleting geom: "{}"'.format(geomdoc['_id']))
                    database.geometry_delete(geomdoc['_id'])


                # add new geoms
                for oldgeomdoc in list(database.db[DBCOLLECTIONS.MCGEOM].find({'image_id': image_id})):
                    try:
                        nbeams = next(database.db[DBCOLLECTIONS.BEAMPHOTON].aggregate([
                            {'$match': {'geom_id': oldgeomdoc['_id']}},
                            {'$count': 'nbeams'}
                        ]))['nbeams']

                        if nbeams is None or nbeams<=0:
                            raise StopIteration
                    except StopIteration as e:
                        print("Geometry \"{}\" is empty, deletion is possible but not performed now".format(oldgeomdoc['_id']))
                        #  database.geometry_delete(oldgeomdoc['_id'])
                        continue

                    geom_id = database.geometry_insert(image_id=image_id,
                                                       **oldgeomdoc['coordsys'],
                                                       )
                    geomdoc = database.get_doc(DBCOLLECTIONS.MCGEOM, geom_id)
                    database.db[DBCOLLECTIONS.MCGEOM].update_one(
                        filter={'_id': oldgeomdoc['_id']},
                        update={"$set": {'procstatus.message': 'Corrupt: flipped sim mcgeom'}}
                    )
                    gfd.write('{!s}\n'.format(oldgeomdoc['_id']))

                    # create new beams
                    for beamdoc in database.db[DBCOLLECTIONS.BEAMPHOTON].find(
                        {'geom_id': oldgeomdoc['_id']}
                    ):
                        if beamdoc is not None:
                            structure_id = beamdoc['structure_id']
                            print("  Structure id: {}".format(structure_id))
                            break
                    assert structure_id is not None

                    structuredoc = next((x for x in database.db[DBCOLLECTIONS.IMAGES].find_one({'_id': image_id})['structures'] if str(x['_id'])==str(structure_id)))
                    actual_isocenter = structuredoc['centroid']
                    iso_shift = (10, 10, 20) # allowable shift from centroid [units: mm]

                    for angle in np.random.uniform(0, 2*math.pi, size=40):
                        beam_id = database.beam_insert(
                            geom_id=geomdoc['_id'],
                            structure_id=structure_id,
                            angle_gantry=angle,
                            angle_couch=0,
                            angle_coll=0,
                            sad=1000,
                            fmapdims=(40,40),
                            beamletspacing=(5,5),
                            beamletsize=(5,5),
                            particletype=PARTICLETYPE.PHOTON,
                            energy='6MV',
                            isocenter=[actual_isocenter[ii] + numpy.random.uniform(-iso_shift[ii], iso_shift[ii]) for ii in range(3)],
                            beamlets={'random-count': 10},
                        )

                        # add simulations
                        beamdoc = database.get_doc(DBCOLLECTIONS.BEAMPHOTON, beam_id)
                        for subbeam in beamdoc['beamlets']:
                            for num_particles in [500, 1000, 2000, 5000, 1e5]:
                                database.simulation_insert(beam_id=beamdoc['_id'],
                                                           subbeam_id=subbeam['_id'],
                                                           vartype=VARTYPE.LOW if num_particles>=1e5 else VARTYPE.HIGH,
                                                           num_particles=num_particles,
                                                           magnetic_field=[0,0,1.5,'tesla'],
                                                           num_runs=5)

        @menucommand
        def add_more_training_data(self, parser):
            image_uids = []
            for imdoc in database.db[DBCOLLECTIONS.IMAGES].find({'doi': {'$not': {"$in": ['HN010', 'HN011']}}}):
                print(imdoc['doi'], imdoc['_id'])
                image_uids.append(imdoc['_id'])

            print()
            for image_id in image_uids:
                print("Image: {}".format(image_id))
                geomdoc = database.db[DBCOLLECTIONS.MCGEOM].find_one({'image_id': ObjectId(image_id)})

                # create new beams for existing structure
                for beamdoc in database.db[DBCOLLECTIONS.BEAMPHOTON].find(
                    {'geom_id': geomdoc['_id']}
                ):
                    if beamdoc is not None:
                        structure_id = beamdoc['structure_id']
                        print("  Structure id: {}".format(structure_id))
                        break
                assert structure_id is not None

                structuredoc = next((x for x in database.db[DBCOLLECTIONS.IMAGES].find_one({'_id': image_id})['structures'] if str(x['_id'])==str(structure_id)))
                actual_isocenter = structuredoc['centroid']
                iso_shift = (4, 4, 4) # allowable shift from centroid [units: mm]

                for angle in np.random.uniform(0, 2*math.pi, size=200):
                    beam_id = database.beam_insert(
                        geom_id=geomdoc['_id'],
                        structure_id=structure_id,
                        angle_gantry=angle,
                        angle_couch=0,
                        angle_coll=0,
                        sad=1000,
                        fmapdims=(40,40),
                        beamletspacing=(5,5),
                        beamletsize=(5,5),
                        particletype=PARTICLETYPE.PHOTON,
                        energy='6MV',
                        isocenter=[actual_isocenter[ii] + numpy.random.uniform(-iso_shift[ii], iso_shift[ii]) for ii in range(3)],
                        beamlets={'random-count': 5},
                    )

                    # add simulations
                    beamdoc = database.get_doc(DBCOLLECTIONS.BEAMPHOTON, beam_id)
                    for subbeam in beamdoc['beamlets']:
                        for num_particles in [1000, 5000, 1e5]:
                            database.simulation_insert(beam_id=beamdoc['_id'],
                                                       subbeam_id=subbeam['_id'],
                                                       vartype=VARTYPE.LOW if num_particles>=1e5 else VARTYPE.HIGH,
                                                       num_particles=num_particles,
                                                       magnetic_field=[0,0,1.5,'tesla'],
                                                       num_runs=5)

        @menucommand
        def calculate_average_time(self, parser):
            import scipy.stats
            sample_size = 1000000
            for num_particles in (2000, 5000, 10000, 1e5):
                sim_iter = database.db[DBCOLLECTIONS.SIMULATION].find({
                    "num_particles": num_particles,
                    "procstatus.status": "finished",
                    "magnetic_field.2": 1.5,
                })
                times = np.zeros((sample_size, ))
                for ii, simdoc in enumerate(tqdm(sim_iter, total=sample_size, desc="calculate average time")):
                    if ii >= sample_size:
                        break
                    times[ii] = simdoc['procstatus']['time_elapsed'] / float(simdoc['num_runs'])
                times = times[:ii]

                print('{} particles: {!s} '.format(num_particles, scipy.stats.describe(times)))

        @menucommand
        def delete_fluence_data(self, parser):
            """Delete all occurences of 3dfluence.bin from database"""
            simdocs = database.db[DBCOLLECTIONS.SIMULATION].find({})
            n = 0
            npart = set()
            for simdoc in simdocs:
                n+=1
                npart.add(simdoc['num_particles'])
                for sample in simdoc['samples']:
                    print('sim_id: {!s}  |  sample_id: {!s}  |  '.format(simdoc['_id'], sample['_id']), end='')
                    if sample['fluencefile'] is None:
                        print('already gone')
                        continue
                    try:
                        os.remove(database.dbabspath(sample['fluencefile']))
                        database.db[DBCOLLECTIONS.SIMULATION].update_one(
                            filter={"_id": ObjectId(simdoc['_id'])},
                            update={'$set': {'samples.$[sample].fluencefile': None}},
                            array_filters=[{"sample._id": ObjectId(sample['_id'])}],
                        )
                        print('deleted')
                    except Exception as e:
                        print('error')
                        print(e)

class CMenuRestAPI(CommandMenuBase):
    description = 'Client for interacting with distributed Monte Carlo Dose Calculation network'
    @menucommand
    def image(self, parser):
        """>>Manipulate database 'Image' documents"""
        self.RestImageMenu()

    @menucommand
    def geometry(self, parser):
        """>>Manipulate database 'Geometry' documents"""
        self.RestGeometryMenu()

    @menucommand
    def beam(self, parser):
        """>>Manipulate database 'Beam' documents"""
        self.RestBeamMenu()

    @menucommand
    def simulation(self, parser):
        """>>Manipulate database 'Simulation' documents"""
        self.RestSimulationMenu()

    @menucommand
    def deeplearning(self, parser):
        """>>Actions supporting Deep Learning"""
        self.RestDeepLearningMenu()

    @menucommand
    def add_simulation(self, parser):
        parser.add_argument('--doi', type=str, required=True, help='human friendly plan id')
        parser.add_argument('--nparticles', metavar='N', type=int, nargs='+', required=True, help='Define one or more simulations of "N" primary particles')
        self.args = parser.parse_args()

        p = payloadtypes.RESTReqImageGet()
        p.image_doi = self.args.doi
        imagedoc = validate_rest_response(p.send_request(dsaddr))

        # get geometry for image
        p = payloadtypes.RESTReqGeometryGet()
        p.filter = {'image_id': ObjectId(imagedoc['_id'])}
        geomdocs = validate_rest_response(p.send_request(dsaddr))
        if geomdocs is None or not len(geomdocs):
            raise RuntimeError('no geometry could be found for image with id "{}" (doi: "{}")'.format(imagedoc['_id'], imagedoc['doi']))
        elif len(geomdocs)>1:
            raise RuntimeError('more than one ({}) geometry could be found for image with id "{}" (doi: "{}")'.format(len(geomdocs), imagedoc['_id'], imagedoc['doi']))
        geomdoc = geomdocs[0]

        # get all beams associated with image
        p = payloadtypes.RESTReqBeamPhotonGet()
        p.filter = {'image_id': ObjectId(imagedoc['_id'])}
        beamdocs = validate_rest_response(p.send_request(dsaddr))
        logger.info('Adding simulation tasks for {} beams'.format(len(beamdocs)))

        for beamdoc in beamdocs:
            for nparticles in self.args.nparticles:
                p = payloadtypes.RESTReqSimulationInsert()
                p.beam_id = beamdoc['_id']
                p.vartype = VARTYPE.LOW
                p.num_runs = 1
                p.num_particles = nparticles
                p.magnetic_field = (0,0,0,'tesla')
                p.beamlets = None # add to all active beamlets in beam
                sim_ids = validate_rest_response(p.send_request(dsaddr))['ids']
                logger.info('Successfully added {} simulation tasks for {} particles to beam "{}"'.format(len(sim_ids), nparticles, beamdoc['_id']))

    @menucommand
    def add_simulation_perbeamlet(self, parser):
        """Add new simulations with a common 'tag' and per-beamlet nparticles values"""
        parser.add_argument('--doi', type=str, required=True, help='human friendly plan id')
        parser.add_argument('--nparticles-file', type=str, required=True, help='text file with a one-line nparticles integer for each beamlet')
        parser.add_argument('--tag', type=str, required=True, help='A filterable tag string that allows exportdata to only see these results')
        parser.add_argument('--extra_args', type=str, required=True, help='extra args for geant4')
        self.args = parser.parse_args()

        p = payloadtypes.RESTReqImageGet()
        p.image_doi = self.args.doi
        imagedoc = validate_rest_response(p.send_request(dsaddr))

        # get geometry for image
        p = payloadtypes.RESTReqGeometryGet()
        p.filter = {'image_id': ObjectId(imagedoc['_id'])}
        geomdocs = validate_rest_response(p.send_request(dsaddr))
        if geomdocs is None or not len(geomdocs):
            raise RuntimeError('no geometry could be found for image with id "{}" (doi: "{}")'.format(imagedoc['_id'], imagedoc['doi']))
        elif len(geomdocs)>1:
            raise RuntimeError('more than one ({}) geometry could be found for image with id "{}" (doi: "{}")'.format(len(geomdocs), imagedoc['_id'], imagedoc['doi']))
        geomdoc = geomdocs[0]

        # get all beams associated with image
        p = payloadtypes.RESTReqBeamPhotonGet()
        p.filter = {'image_id': ObjectId(imagedoc['_id'])}
        beamdocs = validate_rest_response(p.send_request(dsaddr))
        logger.info('Adding simulation tasks for {} beams'.format(len(beamdocs)))

        #TODO(qlyu): Load your per-beamlet nparticles values into a list in dose-matrix column order
        nparticles_allbeamlets = []
        with open(self.args.nparticles_file, 'r') as fd:
            for line in fd:
                nparticles_allbeamlets.append(int(line.strip(' \n')))

        nbeamlets = 0
        for beamdoc in beamdocs:
            nbeamlets += len(beamdoc['beamlets'])
        assert len(nparticles_allbeamlets) == nbeamlets

        mark = 0
        for beamdoc in beamdocs:
            for nparticles_thisbeamlet, beamletdoc in zip(
                    nparticles_allbeamlets[mark:mark+len(beamdoc['beamlets'])],
                    sort_beamlets_columnorder(beamdoc)):
                p = payloadtypes.RESTReqSimulationInsert()
                p.beam_id = beamdoc['_id']
                p.vartype = VARTYPE.LOW
                p.num_runs = 1
                p.num_particles = nparticles_thisbeamlet
                p.magnetic_field = (0,0,0,'tesla')
                p.beamlets = {'ids': [str(beamletdoc['_id'])]}
                p.tag = self.args.tag
                if self.args.extra_args=='positron':
                    p.callargs = ["--output-positronannihilation"]

                sim_ids = validate_rest_response(p.send_request(dsaddr))['ids']
                logger.info(
                    'Successfully added a simulation task for {} particles for beamlet={} with tag "{}" to beam "{}"'.format(
                        nparticles_thisbeamlet,
                        beamletdoc['position'],
                        self.args.tag,
                        beamdoc['_id']))
            mark += len(beamdoc['beamlets'])

    @menucommand
    def delete_simulations(self, parser):
        """Delete a set of simulation documents based on a set of filter criteria"""
        parser.add_argument('--doi', type=str, required=True, help='human friendly plan id')
        parser.add_argument('--tag', type=str, default=None, help='A filterable tag string that allows exportdata to only see these results')
        self.args = parser.parse_args()

        p = payloadtypes.RESTReqImageGet()
        p.image_doi = self.args.doi
        imagedoc = validate_rest_response(p.send_request(dsaddr))

        p = payloadtypes.RESTReqSimulationDelete()
        p.filter = {'image_id': ObjectId(imagedoc['_id']),
                    'tag': self.args.tag}
        result = validate_rest_response(p.send_request(dsaddr))
        logger.info("Deleted {} simulation docs matching filter {!s}".format(len(result['deleted_ids']), p.filter))

    @menucommand
    def create_plan(self, parser):
        """Create an entire treatment plan and all required sim tasks"""
        default_nparticles=[1e6]
        parser.add_argument('ctroot', type=str, help='directory')
        parser.add_argument('--doi', type=str, default=None, help='human friendly plan id (for later reference) (default: <ctroot>)')
        parser.add_argument('--config', type=str, default=None, help='Optionally specify a different plan configuration file (default: <ctroot>/config.json)')
        parser.add_argument('--beamlist', type=str, default=None, help='Optionally specify a different beam specification file (default: "<ctroot>/beamlist.txt")')
        parser.add_argument('--nparticles', metavar='N', type=int, default=default_nparticles, nargs='+', help='Define one or more simulations of "N" primary particles (default: {})'.format('['+','.join(['{:.0e}'.format(x) for x in default_nparticles])+']'))
        self.args = parser.parse_args()

        image_id = None
        def cleanup_partial_plan(image_id):
            try:
                p = payloadtypes.RESTReqImageDelete()
                p.image_id = image_id
                response = validate_rest_response(p.send_request(dsaddr))
                logger.info('Successfully cleaned up incomplete entries')
            except Exception as e:
                logger.error('Error while cleaning up partial plan data')
        def exit_gracefully(sig, frame):
            if image_id is not None:
                logger.info('caught ctrl-c, undoing all new database entries and cleaning up first.')
                cleanup_partial_plan(image_id)
            sys.exit(1)
        signal.signal(signal.SIGINT, exit_gracefully)

        # hardcoded beam params (only used if not in config.json)
        default_fmapdims       = (40, 40) # x/y-number of beamlets per beam
        default_beamletspacing = (5, 5)   # x/y-spacing of beamlets [units: mm]
        default_beamletsize    = (5, 5)   # x/y-radius of electron beamlet [units: mm]

        ctroot = self.args.ctroot
        if self.args.config is None:
            configfile = pjoin(ctroot, 'config.json')
        else:
            configfile = self.args.config
        assert os.path.exists(configfile)

        if self.args.doi is None:
            doi = os.path.basename(self.args.ctroot)
        else:
            doi = self.args.doi

        if self.args.beamlist is None:
            beamlistfile = pjoin(ctroot, 'beamlist.txt')
        else:
            beamlistfile = self.args.beamlist

        logger.info('Setting database reference DOI to "{}"'.format(doi))

        # Read files
        dicom_files = dicomutils.find_dicom_files(ctroot, recursive=True)
        logger.info('Found {} ct image files'.format(len(dicom_files['CT'])))
        logger.info('Found {} rtstruct image files'.format(len(dicom_files.get('RTSTRUCT', []))))
        assert len(dicom_files['CT'])>0
        #  assert len(dicom_files['RTSTRUCT'])>0
        image_uid = dicomutils.get_dicom_seriesuid(dicom_files['CT'][1])
        ct_voxelsize = dicomutils.get_dicom_voxelsize(dicom_files['CT'][1])

        valid_config_keys = ['ptv', 'bbox', 'fmap_dims', 'beamlet_spacing', 'beamlet_size',
                             'voxelsize', 'ptv_margin',
                             'sad', 'energy', 'magnetic_field', 'density_type',
                             'particle_type', 'storage_type', 'sparse_threshold',
                             'gps_filename', 'extra_args', 'priority']
        with open(configfile, 'r') as fd:
            config = json.load(fd)
            # replace all dash "-" with underscore "_" to avoid confusion
            for k,v in dict(config).items():
                if '-' in k:
                    del config[k]
                    config[k.replace('-', '_')] = v
            # validate all existing settings
            for k in config.keys():
                if k not in valid_config_keys:
                    raise KeyError('Setting "{}" in config is not a valid option'.format(k))

        # read required settings
        ptvname = config['ptv']
        bbox    = config.get('bbox', None)
        # overwrite default beam params with config params (if exists)
        ptv_margin       = config.get('ptv_margin',       0)
        voxelsize        = config.get('voxelsize',        None)
        fmapdims         = config.get('fmap_dims',        default_fmapdims)
        beamletspacing   = config.get('beamlet_spacing',  default_beamletspacing)
        beamletsize      = config.get('beamlet_size',     default_beamletsize)
        sad              = config.get('sad',              None) # source-axis-distance [units: mm]
        energy           = config.get('energy',           None) # electron beam energy [units: MeV]
        magnetic_field   = config.get('magnetic_field',   [0,0,0]) # magnetic field components [units: tesla]
        mc_density_type  = config.get('density_type',     MCGEOTYPE.BULKDENS)
        particletype     = config.get('particle_type',    PARTICLETYPE.ELECTRON)
        storage_type     = config.get('storage_type',     STORAGETYPE.SPARSE)
        sparse_threshold = config.get('sparse_threshold', 0.0)
        gps_filename     = config.get('gps_filename',     None) # None will use default, set in generate_input
        extra_args       = config.get('extra_args',       [])
        priority         = config.get('priority',         None) # low number is higher priority (enter negative to force immediate compute)

        assert voxelsize is None \
            or (isinstance(voxelsize, (int, float)) and voxelsize>0) \
            or (isinstance(voxelsize, (tuple, list)) and all([v>0 for v in voxelsize]))
        assert len(magnetic_field) == 3
        assert len(fmapdims)       == 2
        assert len(beamletspacing) == 2
        assert len(beamletsize)    == 2

        allowed_density_types = [MCGEOTYPE.BULKDENS, MCGEOTYPE.INTERP]
        if mc_density_type not in allowed_density_types:
            raise ValueError('"density-type" set in config must be one of ({}), not "{}"'.format(', '.join(allowed_density_types), mc_density_type))
        logger.info('Using monte carlo density type "{}"'.format(mc_density_type))
        allowed_storage_types = [STORAGETYPE.DENSE, STORAGETYPE.SPARSE]
        if storage_type not in allowed_storage_types:
            raise ValueError('"storage-type" set in config must be one of ({}), not "{}"'.format(', '.join(allowed_storage_types), storage_type))
        logger.info('Using data storage type "{}"'.format(storage_type))
        if storage_type == STORAGETYPE.SPARSE:
            logger.info('Using sparse storage threshold: {:0.2%} of beamlet maximum'.format(sparse_threshold))
        allowed_particle_types = [PARTICLETYPE.ELECTRON, PARTICLETYPE.PHOTON, PARTICLETYPE.PROTON]
        if particletype not in allowed_particle_types:
            raise ValueError('"particle-type" set in config must be one of ({}), not "{}"'.format(', '.join(allowed_particle_types), particletype))
        logger.info('Using particle type "{}"'.format(particletype))
        if gps_filename is not None:
            logger.info('Using custom particle source definition in file: "{}"'.format(gps_filename))
        logger.info('Using extra dose calculation arguments: {!s}'.format(extra_args))
        if voxelsize is not None:
            if isinstance(voxelsize, (int, float)):
                voxelsize = [voxelsize]*3
            voxelsize = tuple([float(x) for x in voxelsize])
            assert len(voxelsize) == 3
            logger.info('Using custom voxelsize: {!s} mm'.format(voxelsize))
        logger.info('PTV Margin for beamlet selection is {} mm'.format(ptv_margin))

        try:
            # first check for conflicting entry (same image previously added)
            p = payloadtypes.RESTReqImageGet()
            p.image_doi = doi
            response = p.send_request(dsaddr)
            if response['status'] == STATUS.SUCCESS:
                logger.warning('Image with doi "{}" already exists in database. Would you like to overwrite with new plan?'.format(doi))
                resp = input("(y/[n]): ")
                if resp.lower() in ['y', 'yes']:
                    logger.info('Deleting old plan...')
                    # clear all incomplete entries from database and filesystem
                    p = payloadtypes.RESTReqImageDelete()
                    p.image_doi = doi
                    validate_rest_response(p.send_request(dsaddr))
                    logger.info('Done. continuing')
                else:
                    logger.warning('aborted')
                    sys.exit(1)

            p = payloadtypes.RESTReqImageInsert()
            p.doi = doi
            p.files = {
                'ct': [ socketio.pack_file_binary(fname) for fname in dicom_files['CT'] ],
                'rtstruct': [ socketio.pack_file_binary(fname) for fname in dicom_files.get('RTSTRUCT', []) ],
            }
            response = p.send_request(dsaddr)
            if response['status'] == STATUS.FAILURE:
                p = payloadtypes.RESTReqImageGet()
                p.image_uid = image_uid
                response = validate_rest_response(p.send_request(dsaddr))
                logger.info('Successfully read image data "{}"'.format(response['_id']))
                image_id = response['_id']
            else:
                response = response['content']
                logger.info('Successfully ingested data "{}" with database id "{}" (doi: "{}")'.format(ctroot, response['id'], doi))
                image_id = response['id']

            #===============
            if isinstance(ptvname, dict):
                if "maskfile" in ptvname:
                    maskfile = pjoin(ctroot, ptvname['maskfile'])
                    logger.info('Using ptv derived from voxelized mask file "{}"'.format(maskfile))
                    p = payloadtypes.RESTReqStructureInsert()
                    p.image_id = image_id
                    p.name = os.path.splitext(os.path.basename(maskfile))[0]
                    p.mask = socketio.pack_numpy_array(np.load(maskfile).astype(np.int8), p.name+'.npy')
                    response = validate_rest_response(p.send_request(dsaddr))
                    ptv_structure_id = response['id']
                    logger.info('Successfully inserted structure "{}" with database id "{}"'.format(p.name, ptv_structure_id))
                    #===============
                    p = payloadtypes.RESTReqStructureGet()
                    p.image_id = image_id
                    p.structure_id = ptv_structure_id
                    response = validate_rest_response(p.send_request(dsaddr))
                    ptv_structure_doc = response
                else:
                    raise KeyError("PTV spec in config file is not a valid option")
            else:
                # add ptv structure and get centroid/isocenter coordinates
                p = payloadtypes.RESTReqStructureInsert()
                p.image_id = image_id
                p.name = ptvname
                response = validate_rest_response(p.send_request(dsaddr))
                ptv_structure_id = response['id']
                logger.info('Successfully inserted structure "{}" with database id "{}"'.format(ptvname, ptv_structure_id))
                #===============
                p = payloadtypes.RESTReqStructureGet()
                p.image_id = image_id
                p.structure_id = ptv_structure_id
                response = validate_rest_response(p.send_request(dsaddr))
                ptv_structure_doc = response

            ptv_centroid = ptv_structure_doc['centroid'] # units of mm in CT scanner coordinate system
            logger.info('retrieved PTV centroid as default isocenter: {!s}'.format(ptv_centroid))

            #===============
            # add geometry
            if isinstance(bbox, dict):
                if "maskfile" in bbox:
                    maskfile = pjoin(ctroot, bbox['maskfile'])
                    logger.info('Using bbox derived from voxelized mask file "{}"'.format(maskfile))
                    p = payloadtypes.RESTReqStructureInsert()
                    p.image_id = image_id
                    p.name = os.path.splitext(os.path.basename(maskfile))[0]
                    p.mask = socketio.pack_numpy_array(np.load(maskfile), p.name+'.npy')
                    response = validate_rest_response(p.send_request(dsaddr))
                    bbox_structure_id = response['id']
                    logger.info('Successfully inserted structure "{}" with database id "{}"'.format(p.name, bbox_structure_id))
                    #===============
                    p = payloadtypes.RESTReqStructureGet()
                    p.image_id = image_id
                    p.structure_id = bbox_structure_id
                    response = validate_rest_response(p.send_request(dsaddr))
                    dosebbox = response['boundbox']
                elif "x" in bbox:
                    logger.info('Using custom bbox definition')
                    dosebbox = load_bbox(bbox, voxelsize=ct_voxelsize)
                else:
                    raise KeyError("BBOX spec in config file is not a valid option")
            elif isinstance(bbox, str):
                bbox_structure = bbox
                logger.info('Using bounding box of structure "{}" as bbox definition'.format(bbox_structure))
                if bbox_structure != ptvname:
                    p = payloadtypes.RESTReqStructureInsert()
                    p.image_id = image_id
                    p.name = bbox_structure
                    response = validate_rest_response(p.send_request(dsaddr))
                    bbox_structure_id = response['id']
                    logger.info('Successfully inserted structure "{}" with database id "{}"'.format(bbox_structure, bbox_structure_id))
                else:
                    bbox_structure_id = ptv_structure_id
                #===============
                p = payloadtypes.RESTReqStructureGet()
                p.image_id = image_id
                p.structure_id = bbox_structure_id
                response = validate_rest_response(p.send_request(dsaddr))
                dosebbox = response['boundbox']
            logger.info('bounding box: {!s}'.format(dosebbox))

            # override voxelsize for simulation
            if voxelsize is not None:
                doseframe = FrameOfReference(start=dosebbox['start'], spacing=dosebbox['spacing'], size=dosebbox['size'])
                doseframe.changeSpacing(voxelsize)
                dosebbox = {'start': doseframe.start, 'spacing': doseframe.spacing, 'size': doseframe.size}

            p = payloadtypes.RESTReqGeometryInsert()
            p.image_id = image_id
            p.coordsys = dosebbox
            p.geomtype = mc_density_type
            response = validate_rest_response(p.send_request(dsaddr))
            geom_id = response['id']
            logger.info('Successfully inserted geometry with database id "{}"'.format(geom_id))

            #===============
            # add beams
            beams = read_beamlist(beamlistfile)
            if particletype != PARTICLETYPE.PROTON:
                ensure_exclusive_setting(beams, 'energy', energy)
            ensure_exclusive_setting(beams, 'sad', sad)
            # override standard settings
            for beam in beams:
                if beam.isocenter is None:
                    beam.isocenter = ptv_centroid
                beam.particletype   = particletype
                beam.fmapdims       = fmapdims
                beam.beamletspacing = beamletspacing
                beam.beamletsize    = beamletsize
                beam.beamlets       = None # use raytracing to determine active beamlets
                beam.ptv_margin     = ptv_margin
                beam.gps_template   = gps_filename

            p = payloadtypes.RESTReqBeamPhotonInsert()
            p.mlrole = MLROLE.NONE
            p.geom_id = geom_id
            p.structure_id = ptv_structure_id
            p.beams = beams
            response = validate_rest_response(p.send_request(dsaddr))
            beam_ids = response['ids']
            logger.info('Successfully inserted {:d} beams'.format(len(beam_ids)))
            #===============
            nsims_total = 0
            for bb, beam_id in enumerate(beam_ids):
                for nparticles in self.args.nparticles:
                    p = payloadtypes.RESTReqSimulationInsert()
                    p.beam_id = beam_id
                    p.vartype = VARTYPE.LOW
                    p.num_runs = 1
                    p.num_particles = nparticles
                    p.magnetic_field = (*magnetic_field, 'tesla')
                    p.storage_type = storage_type
                    p.sparse_threshold = sparse_threshold
                    p.callargs = extra_args
                    p.priority = priority
                    response = validate_rest_response(p.send_request(dsaddr))
                    sim_ids = response['ids']
                    for sim_id in sim_ids:
                        if logger.getEffectiveLevel() <= logging.DEBUG:
                            logger.debug('Successfully inserted simulation "{}" with {:.0e} particles for beam "{}"'.format(sim_id, nparticles, beam_id))
                    nsims_total += len(sim_ids)
                logger.info('Successfully inserted {:d} simulations for beam "{}" ({:d}/{:d})'.format(len(sim_ids), beam_id, bb+1, len(beam_ids)))
            logger.info('Successfully inserted {:d} simulations across {:d} beams'.format(nsims_total, len(beam_ids)))

        except Exception as err:
            logger.exception('An error occured while creating the new plan. cleaning up...')
            # clear all incomplete entries from database and filesystem
            cleanup_partial_plan(image_id=image_id)
            logger.error('Exiting with error code (1)')
            sys.exit(1)


    class RestImageMenu(CommandMenuBase):
        """Manupulate Database Image Objects"""

        @staticmethod
        def _add_selector_args(parser):
            parser.add_argument('--id', help='Image db id')
            parser.add_argument('--doi', help='Image doi')
            parser.add_argument('--uid', help='Image dicom uid')

        @menucommand
        def insert(self, parser):
            parser.add_argument('doi', help='unique name assigned to this CT record')
            parser.add_argument('ctroot', help="directory containing CT and config files")
            parser.add_argument('--structres', nargs="+", help='valid structure names (from rtstruct file)')
            self.args = parser.parse_args()

            dicom_files = dicomutils.find_dicom_files(self.args.ctroot, recursive=True)
            p = payloadtypes.RESTReqImageInsert()
            p.doi = self.args.doi
            p.structures = self.args.structures
            p.files = {
                'ct': [ socketio.pack_file_binary(fname) for fname in dicom_files['CT'] ],
                'rtstruct': [ socketio.pack_file_binary(fname) for fname in dicom_files['RTSTRUCT'] ],
            }
            response = validate_rest_response(p.send_request(dsaddr))
            logger.info('Successfully ingested data "{}" with new ctid "{}"'.format(self.args.ctroot, response['id']))

        @menucommand
        def get(self, parser):
            self._add_selector_args(parser)
            self.args = parser.parse_args()

            p = payloadtypes.RESTReqImageGet()
            p.image_doi = self.args.doi
            p.image_id = self.args.id
            p.image_uid = self.args.uid
            response = validate_rest_response(p.send_request(dsaddr))
            pprint(response)

        @menucommand
        def update(self, parser):
            raise NotImplementedError()

        @menucommand
        def delete(self, parser):
            self._add_selector_args(parser)
            self.args = parser.parse_args()

            p = payloadtypes.RESTReqImageDelete()
            p.image_doi = self.args.doi
            p.image_id = self.args.id
            p.image_uid = self.args.uid
            response = validate_rest_response(p.send_request(dsaddr))
            logger.info('Deleted image "{}"'.format(response['id']))

        @menucommand
        def insertstructure(self, parser):
            """insert structure document"""
            parser.add_argument('imageid', help='unique id assigned to this CT record')
            parser.add_argument('name', type=str, help='Structure name')
            self.args = parser.parse_args(namespace=self.args)
            p = payloadtypes.RESTReqStructureInsert()
            p.image_id = self.args.imageid
            p.name = self.args.name
            response = validate_rest_response(p.send_request(dsaddr))
            logger.info('Successfully inserted structure "{}" with new id "{}"'.format(self.args.name, response['id']))

    class RestGeometryMenu(CommandMenuBase):
        """Manupulate Database Geometry Objects"""
        def register_addl_args(self, parser):
            parser.add_argument('id', help='Geometry uid')

        @menucommand
        def insert(self, parser):
            parser.add_argument('image_id', help='unique id assigned to this CT record')
            self.args = parser.parse_args()
            p = payloadtypes.RESTReqGeometryInsert()
            p.image_id = self.args.image_id
            p.coordsys = {
                'start':   None,
                'size':    None,
                'spacing': None,
            }
            p.geomtype = MCGEOTYPE.BULKDENS
            validate_rest_response(p.send_request(dsaddr))
            logger.info('Successfully processed geometry insert for ct "{}"'.format(self.args.ctid))

        @menucommand
        def get(self, parser):
            raise NotImplementedError()

        @menucommand
        def update(self, parser):
            raise NotImplementedError()

        @menucommand
        def delete(self, parser):
            raise NotImplementedError()

    class RestBeamMenu(CommandMenuBase):
        """Manupulate Database Beam Objects"""
        def register_addl_args(self, parser):
            parser.add_argument('id', help='Beam uid')

        @menucommand
        def insert(self, parser):
            raise NotImplementedError()

        @menucommand
        def get(self, parser):
            raise NotImplementedError()

        @menucommand
        def update(self, parser):
            raise NotImplementedError()

        @menucommand
        def delete(self, parser):
            parser.add_argument('beam_id', type=str)
            self.args = parser.parse_args(namespace=self.args)

            p = payloadtypes.RESTReqBeamPhotonDelete()
            p.beam_id = self.args.beam_id
            response = validate_rest_response(p.send_request(dsaddr))
            logger.info('deleted beam "{}"'.format(response['id']))

    class RestSimulationMenu(CommandMenuBase):
        """Manupulate Database Simulation Objects"""
        def register_addl_args(self, parser):
            parser.add_argument('id', help='Simulation uid')

        @menucommand
        def insert(self, parser):
            parser.add_argument('--beam_id', required=True, help='beam objectid')
            parser.add_argument('--nparticles', type=int, required=True, help='simulation # particles')
            parser.add_argument('--nruns', type=int, default=1, help='number of independent simulations')
            parser.add_argument('--vartype', choices=[VARTYPE.LOW, VARTYPE.HIGH], default='high', help='noise class')
            self.args = parser.parse_args(namespace=self.args)

            p = payloadtypes.RESTReqSimulationInsert()
            p.beam_id = self.args.beam_id
            p.vartype = self.args.vartype
            p.num_runs = self.args.nruns
            p.num_particles = self.args.nparticles
            response = validate_rest_response(p.send_request(dsaddr))
            logger.info('Successfully added {} {}-var sim docs for beam "{}"'.format(len(response['ids']), self.args.vartype, self.args.beam_id))

        @menucommand
        def get(self, parser):
            raise NotImplementedError()

        @menucommand
        def update(self, parser):
            raise NotImplementedError()

        @menucommand
        def delete(self, parser):
            raise NotImplementedError()

        @menucommand
        def remove_samples(self, parser):
            parser.add_argument('--nkeep', type=int, default=1, help='how many samples to retain for each matched simulation')
            self.args = parser.parse_args(namespace=self.args)

            p = payloadtypes.RESTReqBeamPhotonGet()
            p.filter = {'mlrole': {'$in': ['test']}}
            response = validate_rest_response(p.send_request(dsaddr))
            sim_ids = [sim_id for beam in response for beamlet in beam['beamlets'] for sim_id in beamlet['simulations']]

            for sim_id in sim_ids:
                p = payloadtypes.RESTReqSimulationGet()
                p.sim_id = sim_id
                response = validate_rest_response(p.send_request(dsaddr))
                removable_samples = []
                if len(response['samples']) > self.args.nkeep:
                    removable_samples = [sample['_id'] for sample in response['samples'][self.args.nkeep:]]
                for sample_id in removable_samples:
                    p = payloadtypes.RESTReqSampleDelete()
                    p.sim_id = sim_id
                    p.sample_id = sample_id
                    response = validate_rest_response(p.send_request(dsaddr))
                    logger.info('Removed sample {}'.format(response))

    class RestDeepLearningMenu(CommandMenuBase):
        @menucommand
        def insert_images_and_geometry(self, parser):
            """test all insertions"""
            parser.add_argument('ctroot', type=str, help='directory')
            parser.add_argument('--doi', default='default_doi')
            self.args = parser.parse_args(namespace=self.args)

            ctroot = self.args.ctroot
            doi = self.args.doi

            dicom_files = dicomutils.find_dicom_files(ctroot, recursive=True)
            p = payloadtypes.RESTReqImageInsert()
            p.doi = doi
            p.files = {
            'ct': [ socketio.pack_file_binary(fname) for fname in dicom_files['CT'] ],
            'rtstruct': [ socketio.pack_file_binary(fname) for fname in dicom_files['RTSTRUCT'] ],
            }
            response = p.send_request(dsaddr)
            if response['status'] == STATUS.FAILURE:
                p = payloadtypes.RESTReqImageGet()
                #p.image_uid = dicomutils.get_dicom_seriesuid(dicom_files['CT'][0])
                p.image_doi = doi
                response = validate_rest_response(p.send_request(dsaddr))
                logger.info('Successfully read image data "{}"'.format(response['_id']))
                response['id'] = response['_id']
            else:
                response = response['content']
                logger.info('Successfully ingested data "{}" with new ctid "{}"'.format(ctroot, response['id']))

            ids_info={"image_id": None, "geom_id": None, "structure_id": None}

            #===============
            ctresponse = response
            image_id = ctresponse['id']
            ids_info['image_id']=str(image_id)
            if os.path.isfile(pjoin(ctroot, 'config.json')):
                with open(pjoin(ctroot, 'config.json'), 'r') as fd:
                    jd = json.load(fd)
                    if 'ptv' in jd and jd['ptv'] is not None:
                        structure = jd['ptv']
                    else:
                        raise RuntimeError("Need to specify ptv name in config.json")
            else:
                raise RuntimeError("Cannot find structure name")

            p = payloadtypes.RESTReqStructureInsert()
            p.image_id = image_id
            p.name = structure
            response = p.send_request(dsaddr)
            if response['status'] == STATUS.FAILURE:
                # get existing structure id instead
                p = payloadtypes.RESTReqStructureGet()
                p.image_id = image_id
                p.structure_name = structure
                response = validate_rest_response(p.send_request(dsaddr))
                print(response)
                response['id'] = str(response['_id'])
            else:
                response = response['content']
            logger.info('Successfully inserted structure "{}" with new structure "{}"'.format(structure, response['id']))
            ids_info['structure_id']=str(response['id'])

            #===============
            if os.path.isfile(pjoin(ctroot, 'config.json')):
                with open(pjoin(ctroot, 'config.json'), 'r') as fd:
                    jd = json.load(fd)
                    if 'bbox' in jd and jd['bbox'] is not None:
                        structure = jd['bbox']
                    else:
                        raise RuntimeError("Need to specify body name in config.json")
            p = payloadtypes.RESTReqStructureInsert()
            p.image_id = image_id
            p.name = structure
            response = p.send_request(dsaddr)
            if response['status'] == STATUS.FAILURE:
                # get existing structure id instead
                p = payloadtypes.RESTReqStructureGet()
                p.image_id = image_id
                p.structure_name = structure
                response = validate_rest_response(p.send_request(dsaddr))
                response['id'] = str(response['_id'])
                print(response)
            else:
                response = response['content']
            logger.info('Successfully inserted structure "{}" with new structure "{}"'.format(structure, response['id']))
            bodystructure_id = response['id']
            #===============
            p = payloadtypes.RESTReqStructureGet()
            p.image_id = image_id
            p.structure_id = bodystructure_id
            response = validate_rest_response(p.send_request(dsaddr))
            logger.info('Successfully retrieved structure doc "{}"'.format(bodystructure_id))
            dosebbox = response['boundbox']

            # override voxelsize for simulation
            voxelsize = (2.5, 2.5, 2.5)
            if voxelsize is not None:
                doseframe = FrameOfReference(start=dosebbox['start'], spacing=dosebbox['spacing'], size=dosebbox['size'])
                doseframe.changeSpacing(voxelsize)
                dosebbox = {'start': doseframe.start, 'spacing': doseframe.spacing, 'size': doseframe.size}

            p = payloadtypes.RESTReqGeometryInsert()
            p.image_id = image_id
            p.coordsys = dosebbox
            p.geomtype = MCGEOTYPE.BULKDENS
            response = validate_rest_response(p.send_request(dsaddr))
            geom_id = response['id']
            ids_info['geom_id']=str(geom_id)
            logger.info('Successfully processed geometry insert with id "{} for ct "{}"'.format(geom_id, image_id))

            with open(pjoin(ctroot, 'ids.txt'), 'w') as outfile:
                json.dump(ids_info, outfile)

        @menucommand
        def add_sims_to_doi(self, parser):
            """add more testing simulations tasks to DOI, beams and geometry will be inferred based on selected structure"""
            parser.add_argument('doi', help='DOI identifier for given patient')
            parser.add_argument('--structure', default=None, help='name of structure')
            self.args = parser.parse_args(namespace=self.args)

            # get image_doc
            p = payloadtypes.RESTReqImageGet()
            p.image_doi = self.args.doi
            imagedoc = validate_rest_response(p.send_request(dsaddr))

            # get structure id
            structuredoc = None
            for structure in imagedoc['structures']:
                if self.args.structure is None:
                    structuredoc = structure
                else:
                    if structure['name'] == self.args.structure:
                        structuredoc = structure
                        break

            # get geometry doc
            p = payloadtypes.RESTReqGeometryGet()
            p.filter = {'image_id': ObjectId(imagedoc['_id'])}
            geomdoc = validate_rest_response(p.send_request(dsaddr))[0]

            # get beamdocs
            p = payloadtypes.RESTReqBeamPhotonGet()
            p.filter = {'geom_id': ObjectId(geomdoc['_id']), 'structure_id': ObjectId(structuredoc['_id'])}
            beamdocs = validate_rest_response(p.send_request(dsaddr))

            # insert simulations tasks
            for beamdoc in beamdocs:
                p = payloadtypes.RESTReqSimulationInsert()
                p.beam_id = beamdoc['_id']
                p.vartype = VARTYPE.LOW
                p.magnetic_field = (0, 0, 0, 'tesla')
                p.num_runs = 1
                p.num_particles = 10e6
                p.beamlets = None # raytrace to select beamlets
                response = validate_rest_response(p.send_request(dsaddr))

                for nparticles in [1000, 2000]:
                    p = payloadtypes.RESTReqSimulationInsert()
                    p.beam_id = beamdoc['_id']
                    p.vartype = VARTYPE.HIGH
                    p.magnetic_field = (0, 0, 0, 'tesla')
                    p.num_runs = 1
                    p.num_particles = nparticles
                    p.beamlets = None # all beamlets in each beam
                    response = validate_rest_response(p.send_request(dsaddr))

        @menucommand
        def add_sims_to_beams(self, parser):
            parser.add_argument('beamlist', help='file listing each beamid (one-per-line)')
            parser.add_argument('--nparticles', type=float, help='number of simulation histories')
            parser.add_argument('--nruns', type=float, default=1, help='number of simulation runs')
            parser.add_argument('--magnetic_field', default=1.5, type=float, help='magnetic field strength in Z-direction (unit: Tesla)')
            self.args = parser.parse_args(namespace=self.args)

            beam_ids = []
            with open(self.args.beamlist, 'r') as fd:
                for beam_id in fd:
                    beam_ids.append(beam_id.rstrip('\n'))

            for beamid in beam_ids:
                # insert simulations tasks
                p = payloadtypes.RESTReqSimulationInsert()
                p.beam_id = ObjectId(beamid)
                p.vartype = VARTYPE.LOW if nparticles>=1e5 else VARTYPE.HIGH
                p.magnetic_field = (0, 0, self.args.magnetic_field, 'tesla')
                p.num_runs = self.args.nruns
                p.num_particles = self.args.nparticles
                p.beamlets = None # add to all beamlets
                response = validate_rest_response(p.send_request(dsaddr))

        @menucommand
        def getbeamlist(self, parser):
            parser.add_argument('geom_id', type=str)
            parser.add_argument('--out', '-o', type=str, default=os.path.curdir)
            self.args = parser.parse_args(namespace=self.args)
            os.makedirs(self.args.out, exist_ok=True)

            p = payloadtypes.RESTReqBeamPhotonGet()
            p.filter = {'geom_id': ObjectId(self.args.geom_id), 'mlrole': 'test'}
            beams = validate_rest_response(p.send_request(dsaddr))
            beams.sort(key=lambda b: b['angle_gantry'])
            beamlist_fname = pjoin(self.args.out, 'beamlist_{!s}.txt'.format(self.args.geom_id))
            beamdetail_fname = pjoin(self.args.out, 'beamdetail_{!s}.txt'.format(self.args.geom_id))
            with open(beamdetail_fname, 'w') as bd:
                bd.write('beam_id  angle_gantry  isocenter  nbeamlets\n')
                with open(beamlist_fname, 'w') as bl:
                    for ii, beam in enumerate(beams):
                        logger.debug('beam_{:03d}: {!s}'.format(ii+1, beam['_id']))
                        bl.write('{!s}\n'.format(beam['_id']))
                        bd.write('{!s}  {!s}  {}  {}\n'.format(beam['_id'], beam['angle_gantry'],
                                                                   ['{:0.2f}'.format(x) for x in beam['isocenter']],
                                                                   len(beam['beamlets']),
                                                                   ))
            logger.info('beamlist written to "{}"'.format(beamlist_fname))

        @menucommand
        def insert_testing_tasks(self, parser):
            """test all insertions"""
            parser.add_argument('--geom_id', required=True, type=str)
            parser.add_argument('--structure_id', required=True, type=str)
            self.args = parser.parse_args(namespace=self.args)

            geom_id = self.args.geom_id
            structure_id = self.args.structure_id

            beams = []
            for angle in dicomutils.generate_spaced_beams(7, start=0):
                beam = payloadtypes.PhotonBeam()
                beam.angle_gantry = angle
                beam.angle_couch = 0
                beam.angle_coll = 0
                beam.sad = 1000
                beam.isocenter = None # inherit target structure centroid
                beam.fmapdims = (40,40)
                beam.beamletspacing = (5,5)
                beam.beamletsize = (5, 5)
                beam.beamlets = None # use raytracing to determine active beamlets
                beam.particletype = PARTICLETYPE.PROTON
                beams.append(beam)

            p = payloadtypes.RESTReqBeamPhotonInsert()
            p.mlrole = MLROLE.TEST
            p.geom_id = geom_id
            p.structure_id = structure_id
            p.beams = beams
            response = validate_rest_response(p.send_request(dsaddr))
            logger.info('Successfully processed beam insert for geom "{}" and structure "{}"'.format(geom_id, structure_id))
            beam_ids = response['ids']
            #===============
            for beam_id in beam_ids:
                logger.info('Inserting simulation docs for beam "{}"'.format(beam_id))

                nparticles = 10e6
                p = payloadtypes.RESTReqSimulationInsert()
                p.beam_id = beam_id
                p.vartype = VARTYPE.LOW
                p.num_runs = 1
                p.num_particles = nparticles
                p.magnetic_field = (0, 0, 0, 'tesla')
                p.storage_type = STORAGETYPE.SPARSE
                p.sparse_threshold = 0.0
                response = validate_rest_response(p.send_request(dsaddr))
                logger.info('Successfully added {} low-var sim docs for beam "{}"'.format(len(response['ids']), beam_id))

                for nparticles in [500, 1000, 2000, 5000]:
                    p = payloadtypes.RESTReqSimulationInsert()
                    p.beam_id = beam_id
                    p.vartype = VARTYPE.HIGH
                    p.num_runs = 1
                    p.num_particles = nparticles
                    p.magnetic_field = (0, 0, self.args.magfield, 'tesla')
                    p.storage_type = STORAGETYPE.SPARSE
                    p.sparse_threshold = 0.0
                    response = validate_rest_response(p.send_request(dsaddr))
                    logger.info('Successfully added {} high-var sim docs for beam "{}" with {} particles'.format(len(response['ids']), beam_id, nparticles))

        @menucommand
        def insert_training_tasks(self, parser):
            parser.add_argument('--geom_id', required=True, type=str)
            parser.add_argument('--structure_id', required=True, type=str)
            self.args = parser.parse_args(namespace=self.args)

            geom_id = self.args.geom_id
            structure_id = self.args.structure_id

            p = payloadtypes.RESTReqGeometryGet()
            p.geom_id = self.args.geom_id
            response = validate_rest_response(p.send_request(dsaddr))
            p = payloadtypes.RESTReqStructureGet()
            p.image_id = response['image_id']
            p.structure_id = self.args.structure_id
            response = validate_rest_response(p.send_request(dsaddr))
            actual_isocenter = response['centroid']

            beams = []
            iso_shift = (10, 10, 20) # allowable shift from centroid [units: mm]
            for angle in numpy.random.uniform(0, 2*math.pi, size=40):
                beam = payloadtypes.PhotonBeam()
                beam.angle_gantry = angle
                beam.angle_couch = 0
                beam.angle_coll = 0
                beam.sad = 1000
                beam.isocenter = [actual_isocenter[ii] + numpy.random.uniform(-iso_shift[ii], iso_shift[ii]) for ii in range(3)]
                beam.fmapdims = (40,40)
                beam.beamletsize = (5, 5)
                beam.beamletspacing = (5, 5)
                beam.beamlets = {'random-count': 30}
                beam.particletype = PARTICLETYPE.PROTON
                beams.append(beam)

            p = payloadtypes.RESTReqBeamPhotonInsert()
            p.mlrole = MLROLE.TRAIN
            p.geom_id = geom_id
            p.structure_id = structure_id
            p.beams = beams
            response = validate_rest_response(p.send_request(dsaddr))
            logger.info('Successfully processed beam insert for geom "{}" and structure "{}"'.format(geom_id, structure_id))
            beam_ids = response['ids']
            #===============
            for beam_id in beam_ids:
                logger.info('Inserting simulation docs for beam "{}"'.format(beam_id))

                for nparticles in [10000, 15000, 20000, 25000]:
                    p = payloadtypes.RESTReqSimulationInsert()
                    p.beam_id = beam_id
                    p.vartype = VARTYPE.HIGH
                    p.num_runs = 5
                    p.num_particles = nparticles
                    p.magnetic_field = (0,0,0,'tesla')
                    p.storage_type = STORAGETYPE.SPARSE
                    p.sparse_threshold = 0.0
                    response = validate_rest_response(p.send_request(dsaddr))
                    logger.info('Successfully added {} high-var sim docs for beam "{}" with {} particles'.format(len(response['ids']), beam_id, nparticles))

                nparticles = 10e6
                p = payloadtypes.RESTReqSimulationInsert()
                p.beam_id = beam_id
                p.vartype = VARTYPE.LOW
                p.num_runs = 1
                p.num_particles = nparticles
                p.magnetic_field = (0,0,0,'tesla')
                p.storage_type = STORAGETYPE.SPARSE
                p.sparse_threshold = 0.0
                response = validate_rest_response(p.send_request(dsaddr))
                logger.info('Successfully added {} low-var sim docs for beam "{}" with {} particles'.format(len(response['ids']), beam_id, nparticles))

class ClientMenu(CommandMenuBase):
    description = 'Client for interacting with distributed Monte Carlo Dose Calculation network'

    def run_after_parse(self):
        # setup module logger
        global logger, dsaddr
        logger = log.get_module_logger(__name__, level=self.args.loglevel)
        # configure dataserver address
        dsaddr = (self.args.dsaddr, self.args.dsport)

    def register_addl_args(self, parser):
        log.add_argument_loglevel(parser)
        parser.add_argument('--dsaddr', default=defaults.ds_address[0], help='Dataserver address')
        parser.add_argument('--dsport', default=defaults.ds_address[1], help='Dataserver port')

    @menucommand
    def restapi(self, parser):
        """>>Interact with dataserver through REST API"""
        CMenuRestAPI()

    @menucommand
    def database(self, parser):
        """>>Directly manipulate database"""
        CMenuDatabase()

    def debug(self, parser):
        """>>Debug Commands"""
        CMenuDebug()

    @menucommand
    def status(self, parser):
        """continuously query for simulation task status"""
        parser.add_argument('--csaddr', nargs='+', default=defaults._cs_hosts, help='Computeserver address')
        parser.add_argument('--csport', default=defaults._cs_port, help='Computeserver port')
        parser.add_argument('--rate','-n','-r', type=float, default=None, help='update rate (seconds)')
        self.args = parser.parse_args(namespace=self.args)

        logger.info('Requesting compute server status from {} compute node(s)'.format(len(self.args.csaddr)))
        while True:
            sock = socket.create_connection(dsaddr, 60)
            response = socketio.send_request(sock, {'type': MESSAGETYPE.STATUSREQUEST})
            logger.info('{:50s} '.format('DS ({!s}:{!s}):'.format(*dsaddr))+str(response))
            for csaddr in self.args.csaddr:
                prefix = '{:50s} '.format('CS ({!s}:{!s}):'.format(csaddr, self.args.csport))
                try:
                    sock = socket.create_connection((csaddr, self.args.csport), 60)
                    response = socketio.send_request(sock, {'type': MESSAGETYPE.STATUSREQUEST})
                    logger.info(prefix+str(response))
                except (ConnectionRefusedError, socket.gaierror)  as err:
                    logger.info(prefix+"Failed to connect: "+str(err))
                except Exception as err:
                    logger.info(prefix+"Failed to connect: "+str(err))
            if self.args.rate is None:
                break
            logger.info('---')
            time.sleep(self.args.rate)

    @menucommand
    def dbsummary(self, parser):
        """Summarize the data in the database"""
        self.args = parser.parse_args(namespace=self.args)
        response = validate_rest_response(payloadtypes.SummaryRequest().send_request(dsaddr, timeout=None))
        logger.info('Database Summary:\n'+pformat(response))

    @menucommand
    def plan_status(self, parser):
        parser.add_argument('--sortby', '-s', default='date', choices=['date', 'progress', 'name'], help='Change display sorting')
        parser.add_argument('--invertsort', '-i', action='store_true', help='sort descending')
        parser.add_argument('--field', '-f', action='append', default=[], choices=['tag', 'nparticles'], help="Add an additional field to the table")
        parser.add_argument('--no-truncate', action='store_true', help='don\'t shorten long output')
        self.args = parser.parse_args()

        p = payloadtypes.PlanStatusRequest()
        plans = validate_rest_response(p.send_request(dsaddr, timeout=60))

        def trunc(s, size):
            if self.args.no_truncate:
                return s
            if len(s) > size:
                s = s[:size-3] + '...'
            return s

        # convert datetime from string
        for plan in list(plans):
            if plan is None:
                plans.remove(plan)
                continue
            plan['date_added'] = datetime.strptime(plan['date_added'],
                                                   '%Y-%m-%d_%H:%M:%S')

        # sort results
        if self.args.sortby == 'date':
            sort_key = lambda plan: plan['date_added']
        elif self.args.sortby == 'progress':
            def sort_key(plan):
                num_total_sims = plan['sim-status']['pending'] + \
                                 plan['sim-status']['finished'] + \
                                 plan['sim-status']['failed']
                return plan['sim-status']['finished'] / num_total_sims
        elif self.args.sortby == 'name':
            sort_key=lambda plan: plan['doi']
        plans.sort(key=sort_key, reverse=self.args.invertsort)

        def format_particle_count(x):
            if x // 1e3 == 0:
                return '{:0.0f}'.format(x)
            elif x // 1e6 == 0:
                return '{:0.0f}k'.format(x/1e3)
            elif x // 1e9 == 0:
                return '{:0.0f}m'.format(x/1e6)
            else:
                return '{:0.0f}g'.format(x/1e9)

        def format_progress(finished, total):
            if finished == total:
                perc_complete = 1.0
            else:
                perc_complete = min(0.99, float(plan['sim-status']['finished']+plan['sim-status']['skipped'])/num_total_sims)
            return '{:5d} / {:5d} ({:4.0%})'.format(finished, total, perc_complete)

        Field = namedtuple('Field', ('name', 'size', 'formatter'))
        Field.__new__.__defaults__ = (None,)*len(Field._fields)
        fields = [
            Field('Plan Name',    60),
            Field('Date Created', 16, lambda d: d.strftime('%Y-%m-%d %H:%M')),
            Field('Particle',     8),
            Field('Density Type', 12),
            Field('# Beams',      7, lambda l: str(l)),
        ]

        extra_fields = list(set(self.args.field))
        if not extra_fields:
            extra_fields.append('nparticles')
        if 'nparticles' in extra_fields:
            fields.append(Field('# Particles', 20, lambda c: ', '.join([format_particle_count(x) for x in sorted(c)])))
        if 'tag' in extra_fields:
            fields.append(Field('Tags', 30, lambda l: ' '.join((str(x) for x in l))))

        fields += [
            Field('Completed Sims', len(format_progress(1,1)), format_progress),
            Field('Message',        20)
        ]

        fieldmap = {}
        for field in fields:
            fieldmap[field.name] = field

        def dyformat(size, *args, formatter=None, **kwargs):
            fmt = '{{!s:{size}}}'.format(size=size if size else '')
            if callable(formatter):
                s = formatter(*args, **kwargs)
            else:
                s = ' '.join(args) + ' '.join(('{!s}={!s}'.format(a,b) for a,b in kwargs.items()))
            return trunc(fmt.format(s), size)

        borderchar = '-'
        space = ' '*2

        # print header
        header = ''
        border = ''
        for f in fields:
            header += dyformat(f.size, f.name) + space
            border += borderchar*f.size + space
        print(header)
        print(border)

        def format_field(key, *args, **kwargs):
            field = fieldmap[key]
            return dyformat(field.size, *args, formatter=field.formatter, **kwargs) + space

        for plan in plans:
            msg = ''
            if plan['sim-status']['failed'] >0:
                msg = '{:d} sims have failed. Restart to retry or investigate the issue'.format(plan['sim-status']['failed'])
            num_total_sims = plan['sim-status']['pending'] + \
                             plan['sim-status']['finished'] + \
                             plan['sim-status']['skipped'] + \
                             plan['sim-status']['failed']

            row = ''
            row += format_field("Plan Name",    plan['doi'])
            row += format_field("Date Created", plan['date_added'])
            row += format_field("Particle",     plan['particle-type'])
            row += format_field("Density Type", plan['density-type'])
            row += format_field("# Beams",      plan['num-beams'])
            try:
                row += format_field("# Particles",  plan['particle-counts'])
            except KeyError:
                pass
            try:
                row += format_field("Tags",  plan['tags'])
            except KeyError:
                pass
            row += format_field("Completed Sims", plan['sim-status']['finished']+plan['sim-status']['skipped'], num_total_sims)
            row += format_field("Message", msg)
            print(row)

if __name__ == '__main__':
    ClientMenu()
