#!/usr/bin/env python1
import sys, os
import traceback
from os.path import join as pjoin
import shutil
import signal
import socket
import time
import datetime
from datetime import timedelta
import argparse
import random
import queue
import heapq
from queue import PriorityQueue, Queue
import multiprocessing
from functools import total_ordering
from collections import defaultdict, namedtuple

from bson import ObjectId

import defaults
import payloadtypes
import restactions
from workertypes import StoppableThread, WorkerStopped, ScheduledTask
from api_enums import (MESSAGETYPE, STATUS, PROCSTATUS,
                       DBCOLLECTIONS, MCGEOTYPE)
import geometry
import parse
import socketio
import database
import log
from log import logtask

import numpy as np  # QL

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', type=str, default="db_data", help="set data root directory")
parser.add_argument('--bindaddr', type=str, help='address on which this service is listening',
                    default=defaults.ds_address[0])
parser.add_argument('--bindport', type=int, help='port on which this service is listening',
                    default=defaults.ds_address[1])
parser.add_argument('--test', action='store_true', help='setup test database and apply ops to that instead')
parser.add_argument('--nocompute', action='store_true', help='Only expose the REST API without issuing compute tasks')
parser.add_argument('--cleandb', action='store_true', help='check all data and clean database')
parser.add_argument('--regen_gps', action='store_true', help='Regenerate all GPS file and reset sims')
parser.add_argument('--dryrun', action='store_true', help='simulate cleanup process')
parse.register_db_args(parser)
parse.register_computeaddress_args(parser)
log.add_argument_loglevel(parser)
args = parser.parse_args()

logger = log.get_module_logger(__name__, level=args.loglevel)

class GeometryGenerator(StoppableThread):
    q_geometry = Queue()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def stop(self):
        #  logger.debug('Stopping worker process "{}"'.format(self.name))
        super().stop()

    def _cleanup_leftovers(self, clean_all=False):
        geomdocs = list(database.get_docs_by_status(DBCOLLECTIONS.MCGEOM, PROCSTATUS.FAILED))
        if clean_all:
            geomdocs += list(database.get_docs_by_status(DBCOLLECTIONS.MCGEOM, PROCSTATUS.QUEUED)) + \
                        list(database.get_docs_by_status(DBCOLLECTIONS.MCGEOM, PROCSTATUS.INPROGRESS))
        logger.info('Cleaning up {} failed and leftover simulations from previous server instance'.format(len(geomdocs)))
        for geomdoc in geomdocs:
            database.update_doc_status(DBCOLLECTIONS.MCGEOM, geomdoc['_id'], PROCSTATUS.PENDING)

    def _populate_queue(self):
        geomdocs = list(database.get_docs_by_status(DBCOLLECTIONS.MCGEOM, PROCSTATUS.PENDING))
        logger.info('Filling geometry queue with {} new jobs'.format(len(geomdocs)))
        for geomdoc in geomdocs:
            geom_id = geomdoc['_id']
            database.update_doc_status(DBCOLLECTIONS.MCGEOM, geom_id, PROCSTATUS.QUEUED)
            GeometryGenerator.q_geometry.put(geomdoc['_id'])

    @staticmethod
    def _mp_calculate_geometry(geomfile, vol, spacing, bulk_density=False, isoshift=np.array([0,0,0])):  # QL
#    def _mp_calculate_geometry(geomfile, vol, spacing, bulk_density=False):
        # need to force new this new thread to handle SIGTERM normally
        def sighandler(signum, stackframe):
            sys.exit(128+int(signum))
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signal.SIGINT, sighandler)

        geometry.generate_geometry(geomfile, vol, spacing, bulk_density=bulk_density, isoshift=isoshift)  # QL
    #    geometry.generate_geometry(geomfile, vol, spacing, bulk_density=bulk_density)

    def _calculate_geometry(self, geom_id):
        """generate a single geometry file for the beam and settings packed in 'task' """
        geomfile = pjoin(database.build_datapath_geom(geom_id), 'mcgeo.txt')
        os.makedirs(os.path.dirname(geomfile), exist_ok=True)
        vol, spacing = database.get_geometry_volume(geom_id)

        geomdoc = database.get_geometry_doc(geom_id)
        use_bulk_density = (geomdoc.get('geomtype', None) == MCGEOTYPE.BULKDENS)

        # lookup list of beams assigned to this geometry, then grab the iso from the first beam
        beamdocs = list(database.db[DBCOLLECTIONS.BEAMPHOTON].find({'geom_id': ObjectId(geomdoc['_id'])}))
        isocenter = beamdocs[0]['isocenter'] # arranged as (x, y, z) in millimeters

        gcs = geomdoc['coordsys'] # QL
        start = np.array(gcs['start']) # QL
        size = np.array(gcs['size']) # QL
        spacing = np.array(gcs['spacing']) # QL

        ctx = multiprocessing.get_context('spawn')
        isopos = np.array(isocenter)  # QL
        centerofvolume = 0.5*np.multiply(size-1, spacing) # QL
        isoshift = np.subtract(np.subtract(isopos,start), centerofvolume) # QL
        proc = ctx.Process(target=self._mp_calculate_geometry, args=(geomfile, vol, spacing, use_bulk_density, isoshift)) # QL

        try:
            proc.start()
            while True:
                proc.join(0.1)
                if proc.exitcode is not None:
                    if proc.exitcode != 0:
                        raise Exception()
                    break
                elif self.is_stopped():
                    raise WorkerStopped()
            database.register_generated_geometry(geom_id, geomfile)
        except Exception as e:
            if isinstance(e, WorkerStopped):
                logger.warning('Task was stopped early ({})'.format(geom_id))
            else:
                logger.warning('Error while running task ({})'.format(geom_id))
            raise
        finally:
            proc.terminate()
            proc.join()

    def run(self):
        def task_cb_cleanup():
            self._setidle(False)
            self._cleanup_leftovers(clean_all=True)
            # TODO: validate entries in queue every so often (in case a task has since been deleted but still in queue)
            self._setidle()
        def task_cb_enqueue():
            self._setidle(False)
            self._populate_queue()
            self._setidle()
        scheduled_tasks = [
            ScheduledTask(task_cb_cleanup, timedelta(minutes=30)),
            ScheduledTask(task_cb_enqueue, timedelta(minutes=1)),
        ]

        geom_id = None
        while True:
            try:
                # populate queue
                if self.workernum == 0:
                    # check/run scheduled_tasks
                    for stask in scheduled_tasks:
                        stask.update()
            except Exception as e:
                logger.exception("Error while running scheduled tasks")
            #---------------
            try:
                if self.is_stopped():
                    raise WorkerStopped()
                if self.workernum == 0:
                    # worker 0 dedicated to queue filling/cleanup
                    continue
                try:
                    geom_id = GeometryGenerator.q_geometry.get(False)
                    if self.is_stopped() or geom_id is None:
                        raise WorkerStopped()
                except queue.Empty as e:
                    continue
                self._setidle(False)
                logger.info('Processing Geometry task ({})'.format(geom_id))
                database.update_doc_status(DBCOLLECTIONS.MCGEOM, geom_id, PROCSTATUS.INPROGRESS)
                start_time = time.perf_counter()
                self._calculate_geometry(geom_id)
                elapsed_time_seconds = time.perf_counter()-start_time
                database.update_doc_status(DBCOLLECTIONS.MCGEOM, geom_id, PROCSTATUS.FINISHED, time_elapsed=elapsed_time_seconds)
                logger.info('Finished Geometry task ({}) in {} seconds'.format(geom_id, elapsed_time_seconds))
                self.q_geometry.task_done()
            except WorkerStopped as e:
                self.q_geometry.task_done()
                logger.debug('exiting process ({})'.format(self.name))
                break
            except Exception as e:
                # protect worker in case any exceptions slip through
                geom_id = 'unknown' if not geom_id else geom_id
                logger.exception("Severe error has occured. Geometry task has been lost ({})".format(geom_id))
                try:
                    database.update_doc_status(DBCOLLECTIONS.MCGEOM, geom_id, PROCSTATUS.FAILED)
                except: pass
                self.q_geometry.task_done()
            finally:
                time.sleep(1)


class SimulationScheduler(StoppableThread):
    sim_queue = PriorityQueue() # of SimulationScheduler.SimTask() objects
    compute_hosts = []

    Host = namedtuple("Host", ('hostname', 'addr'))

    class SimulationRequestSkipped(Exception):
        pass

    @total_ordering
    class SimTask():
        def __init__(self, sim_id, priority=9999):
            """default to lowest possible priority (highest number), lower priority == higher importance"""
            self.sim_id = sim_id
            self.priority = priority

        def __lt__(self, other):
            return (self.priority < other.priority)

        def __eq__(self, other):
            return (self.priority == other.priority and \
                    self.sim_id   == other.sim_id)


    def __init__(self, *args, compute_hosts=None, **kwargs):
        super().__init__(*args, **kwargs)

        if self.workernum == 0:
            if compute_hosts is not None:
                addrs = socketio.get_hosts_by_dns(compute_hosts)
            else:
                addrs = '127.0.0.1'

            for addr in addrs:
                hostname = socketio.get_hostname_by_addr(addr).split('.', 1)[0]
                self.compute_hosts.append(SimulationScheduler.Host(hostname, addr))

            logger.info('Initialized SimulationScheduler with hosts [{!s}]'.format(', '.join([str(x.hostname) for x in self.compute_hosts])))

    def _cleanup_leftovers(self, clean_all=False):
        simdocs = list(database.get_docs_by_status(DBCOLLECTIONS.SIMULATION, PROCSTATUS.FAILED))
        simdocs += list(database.get_corrupted_simulation_docs())
        if clean_all:
            simdocs += list(database.get_docs_by_status(DBCOLLECTIONS.SIMULATION, PROCSTATUS.QUEUED)) + \
                       list(database.get_docs_by_status(DBCOLLECTIONS.SIMULATION, PROCSTATUS.INPROGRESS))
        logger.info('Cleaning up {} leftover and corrupted simulations from previous server instance'.format(len(simdocs)))
        for simdoc in simdocs:
            database.update_doc_status(DBCOLLECTIONS.SIMULATION, simdoc['_id'], PROCSTATUS.PENDING)

    def _populate_queue(self):
        simdocs = list(database.get_docs_by_status(DBCOLLECTIONS.SIMULATION, PROCSTATUS.PENDING,
                                                   projection=['_id']))
        skipped = 0
        logger.info('Checking status of {} simulation tasks'.format(len(simdocs)))
        for simdoc in list(simdocs):
            if not database.check_sim_ready(simdoc['_id']):
                skipped += 1
                simdocs.remove(simdoc)
        if skipped:
            logger.info('Skipped {} simulation jobs because prerequisites were not met'.format(skipped))

        logger.info('Filling simulation queue with {} new jobs'.format(len(simdocs)))
        for simdoc in simdocs:
            sim_id = simdoc['_id']
            database.update_doc_status(DBCOLLECTIONS.SIMULATION, sim_id, PROCSTATUS.QUEUED)

            simtask = self.SimTask(simdoc['_id'])
            self.sim_queue.put(simtask)

    def _populate_queue_by_priority(self):
        """populate simluation queue, prioritizing simulation tasks with lowest priority setting"""
        simdocs = list(database.get_docs_by_status(DBCOLLECTIONS.SIMULATION, PROCSTATUS.PENDING,
                                                   projection=['_id', 'priority']))
        skipped = 0
        logger.info('Checking status of {} simulation tasks'.format(len(simdocs)))
        for simdoc in list(simdocs):
            if not database.check_sim_ready(simdoc['_id']):
                skipped += 1
                simdocs.remove(simdoc)
        if skipped:
            logger.info('Skipped {} simulation jobs because prerequisites were not met'.format(skipped))

        logger.info('Filling simulation queue with {} new jobs'.format(len(simdocs)))
        for simdoc in simdocs:
            sim_id = simdoc['_id']
            database.update_doc_status(DBCOLLECTIONS.SIMULATION, sim_id, PROCSTATUS.QUEUED)

            try:
                priority = int(simdoc['priority'])
            except:
                priority = 9999
            simtask = self.SimTask(simdoc['_id'], priority=priority)
            self.sim_queue.put(simtask)

    def _populate_queue_by_progress(self):
        """populate simluation queue, prioritizing simulation tasks belonging to plans with furthest progress.

        For example, if two plans are not complete (one at 30%, one at 70%) place all tasks for the plan with
        higher fraction of its tasks at the top of the queue, followed by plan with next highest progress
        """
        simdocs = list(database.get_docs_by_status(DBCOLLECTIONS.SIMULATION, PROCSTATUS.PENDING,
                                                   projection=['_id', 'image_id', 'num_particles', 'priority']))
        skipped = 0
        logger.info('Checking status of {} simulation tasks'.format(len(simdocs)))
        for simdoc in list(simdocs):
            if not database.check_sim_ready(simdoc['_id']):
                skipped += 1
                simdocs.remove(simdoc)
        if skipped:
            logger.info('Skipped {} simulation jobs because prerequisites were not met'.format(skipped))

        logger.info('Filling simulation queue with {} new jobs'.format(len(simdocs)))

        # sort sims into groups by parent image and num_particles
        # key is (image_id, nparticles) which defines a single "Plan"
        sim_groups = defaultdict(list)
        for simdoc in simdocs:
            groupkey = ( str(simdoc['image_id']), simdoc['num_particles'] )
            sim_groups[groupkey].append(simdoc)

        # get progress for each image
        progress = []
        for groupkey in sim_groups.keys():
            (image_id, num_particles) = groupkey
            # get progress
            npending = database.get_number_of_docs_for_image_by_status(
                image_id, [PROCSTATUS.FAILED, PROCSTATUS.INPROGRESS, PROCSTATUS.QUEUED, PROCSTATUS.PENDING],
                num_particles=num_particles,
            )
            nfinished = database.get_number_of_docs_for_image_by_status(
                image_id, [PROCSTATUS.FINISHED], num_particles=num_particles,
            )
            progress_for_group = float(nfinished) / (npending + nfinished)
            progress.append((progress_for_group, groupkey))
        progress.sort(reverse=True) # sort images with most progress to front

        # assign priorities
        priority = -len(progress)
        for (progress_for_image, groupkey) in progress:
            for simdoc in sim_groups[groupkey]:
                sim_id = simdoc['_id']
                database.update_doc_status(DBCOLLECTIONS.SIMULATION, sim_id, PROCSTATUS.QUEUED)

                simtask = self.SimTask(simdoc['_id'], priority=priority)
                self.sim_queue.put(simtask)
            priority += 1

    def _check_queue_contents(self, limit=100):
        nchecked = 0
        while (limit is None or nchecked<limit):
            try:
                task = self.sim_queue.get(False, timeout=10)
                nchecked += 1
            except queue.Empty:
                break

            database.update_doc_status(DBCOLLECTIONS.SIMULATION, task.sim_id, PROCSTATUS.PENDING)
            simdoc = database.db[DBCOLLECTIONS.SIMULATION].find_one({'_id': ObjectId(task.sim_id)})
            imagedoc = database.db[DBCOLLECTIONS.IMAGES].find_one({'_id': ObjectId(simdoc['image_id'])})
            print('{:5d})  {:40s}  {:5d} {!s}'.format(nchecked, imagedoc['doi'], (task.priority if task.priority<9999 else 9999), task.sim_id))

    def send_simulation_request(self, sim_id, timeout=10):
        """Sends a simulation request (payloadtypes.SimInstruction)

        SimInstruction payloads support multiple payloadtypes.SimulationConfig objects
        for a single subbeam_id, but this function currently only supports sending
        one SimulationConfig per SimInstruction.
        """
        # construct valid payload
        #TODO: Support sending multiple SimulationConfig for each SimInstruction
        #      Accept a list of sim_id's to permit bundled computation.
        #      computeserver.py already supports bundling.
        payload = database.generate_simulation_payload(sim_id)
        logger.debug('Prepared simulation payload with {:d} configurations for sim id "{!s}"'.format(
            len(payload.simulations), sim_id))

        # check if any sims can be skipped. remove from payload if so
        if payload.simulations:
            for simconfig in list(payload.simulations):
                if simconfig.num_particles <= 0:
                    database.process_skipped_simulation(simconfig.id)
                    payload.simulations.remove(simconfig)
        if not payload.simulations:
            raise SimulationScheduler.SimulationRequestSkipped

        payload.reply_host = args.bindaddr
        payload.reply_port = args.bindport
        for sim in payload.simulations:
            database.update_doc_status(DBCOLLECTIONS.SIMULATION, sim.id, PROCSTATUS.INPROGRESS)

        starttime = time.perf_counter()
        port_compute = args.computeport
        while True:
            ipidx = random.randrange(len(self.compute_hosts))
            if timeout and timeout > 0 and time.perf_counter()-starttime > timeout:
                raise OSError('Timeout reached while trying to submit processing request')
            try:
                hostname, ip_compute = self.compute_hosts[ipidx]
                logger.debug('sending simulation task to host "{}" ({}:{})'.format(hostname, ip_compute, port_compute))
                response = payload.send_request((ip_compute, port_compute), timeout=None, connection_timeout=1)
                if response['status'] == STATUS.SUCCESS:
                    break
                else:
                    logger.debug('simulation task was rejected by host "{}" ({}:{}). most likely due to a '
                                 'full simulation queue on the host'.format(hostname, ip_compute, port_compute))

            except (socket.timeout, ConnectionRefusedError, ConnectionResetError):
                logger.debug('timeout while trying to connect to "{}:{}"'.format(ip_compute, port_compute))
                ipidx = (ipidx+1)%len(self.compute_hosts)
                time.sleep(1)
        return (ip_compute, port_compute)

    def run(self):
        """setup queue of simulation tasks and work through one by one"""
        def task_cb_cleanup():
            self._setidle(False)
            self._cleanup_leftovers(clean_all=False)
            self._setidle()
        def task_cb_enqueue():
            self._setidle(False)
            self._populate_queue_by_priority()
            self._setidle()
        scheduled_tasks = [
            ScheduledTask(task_cb_cleanup, timedelta(minutes=10)),
            ScheduledTask(task_cb_enqueue, timedelta(minutes=1)),
        ]

        # Cleanup failed and mislabeled tasks from last run
        if self.workernum == 0:
            self._setidle(False)
            self._cleanup_leftovers(clean_all=True)
            self._setidle()

        while True:
            try:
                # populate queue
                if self.workernum == 0:
                    # check/run scheduled tasks
                    for stask in scheduled_tasks:
                        stask.update()
            except Exception as e:
                logger.exception("Error while running scheduled tasks")
            #---------------
            try:
                if self.is_stopped():
                    raise WorkerStopped()
                try:
                    task = self.sim_queue.get(False)
                    sim_id = task.sim_id
                    if self.is_stopped() or sim_id is None:
                        raise WorkerStopped()
                except queue.Empty as e:
                    continue
                self._setidle(False)
                logger.info('Sending simulation request for {!s}'.format(sim_id))
                # TODO: Bundle multiple simtasks if they share a beamlet_id
                self.send_simulation_request(sim_id, timeout=None)
                self.sim_queue.task_done()

            except SimulationScheduler.SimulationRequestSkipped:
                logger.debug('No SimulationConfigs to send. Skipping')
            except WorkerStopped as e:
                logger.debug('exiting thread {}'.format(self.getName()))
                break
            except Exception as e:
                logger.exception("Error while trying to send simulation request")
                try: database.update_doc_status(DBCOLLECTIONS.SIMULATION, sim_id, PROCSTATUS.FAILED)
                except: pass
                self.sim_queue.task_done()


class ConnectionHandler(StoppableThread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def handle_simreport(simreport):
        assert(isinstance(simreport, payloadtypes.SimReport))
        for sim in simreport.simulations:
            try:
                outputdir = pjoin(database.DATASTORE.SIMDATA, simreport.beam_id, simreport.subbeam_id, sim.id)
                database.process_completed_simulation(sim, simreport.beam_id, simreport.subbeam_id, outputdir, simreport.status, simreport.message)
            except:
                logger.exception("Error while registering simulation report. Results are lost")

    @staticmethod
    def handle_rest_request(payload):
        """switchboard to all possible REST API actions"""
        assert isinstance(payload, payloadtypes.RESTReqBase)
        rest_action = 'rest_{doctype}_{reqtype}'.format(doctype=payload.doctype, reqtype=payload.reqtype)
        logger.info('executing REST action: "{}"'.format(rest_action))
        try:
            func = getattr(restactions, rest_action)
        except AttributeError as e:
            raise NotImplementedError('REST action "{}" is not yet implemented'.format(rest_action))
        content = socketio.make_json_friendly(func(payload))
        return content

    @staticmethod
    def handle_plan_status_request(payload):
        plans = []
        for imagedoc in database.db[DBCOLLECTIONS.IMAGES].find():
            geomdoc = database.db[DBCOLLECTIONS.MCGEOM].find_one({'image_id': imagedoc['_id']})
            if geomdoc is None:
                raise RuntimeError("Corruption detected in database. Please run database cleanup and try again")
            beamdoc = database.db[DBCOLLECTIONS.BEAMPHOTON].find_one({'image_id': imagedoc['_id']})
            try:
                num_beams = next(database.db[DBCOLLECTIONS.BEAMPHOTON].aggregate([
                    {'$match': {'image_id': imagedoc['_id']}},
                    {'$count': 'count'}
                ]))['count']
            except StopIteration: num_beams = 0
            try:
                particle_counts = database.db[DBCOLLECTIONS.SIMULATION].find({"image_id": imagedoc['_id']}).distinct('num_particles')
            except:
                particle_counts = []

            num_pending_sims = database.get_number_of_docs_for_image_by_status(
                imagedoc['_id'], [PROCSTATUS.PENDING,
                                  PROCSTATUS.INPROGRESS,
                                  PROCSTATUS.QUEUED]
            )
            num_finished_sims = database.get_number_of_docs_for_image_by_status(
                imagedoc['_id'], PROCSTATUS.FINISHED
            )
            num_failed_sims = database.get_number_of_docs_for_image_by_status(
                imagedoc['_id'], PROCSTATUS.FAILED
            )
            num_skipped_sims = database.get_number_of_docs_for_image_by_status(
                imagedoc['_id'], PROCSTATUS.SKIPPED
            )
            try:
                taglist = database.db[DBCOLLECTIONS.SIMULATION].find({"image_id": imagedoc['_id']}).distinct('tag')
            except:
                taglist = []
            try:
                plan = {
                    'doi': imagedoc['doi'],
                    'date_added': imagedoc['date_added'],
                    'particle-type': beamdoc['particletype'],
                    'density-type': geomdoc['geomtype'],
                    'num-beams': num_beams,
                    'particle-counts': particle_counts,
                    'sim-status': {'pending': num_pending_sims,
                                   'finished': num_finished_sims,
                                   'failed': num_failed_sims,
                                   'skipped': num_skipped_sims},
                    'tags': taglist,
                }
            except:
                plan = None
            plans.append(plan)
        return plans

    def run(self):
        """handle incoming notifications from finished tasks"""
        while True:
            try:
                self._setidle()
                sock = q_connections.get()
                if sock is None:
                    logger.debug('exiting thread {}'.format(self.getName()))
                    break

                self._setidle(False)
                payload = socketio.receive_all(sock)
                try:
                    # Validate payload data before handling
                    payload = payloadtypes.BasePayload.fromdict(payload)
                except TypeError as e:
                    pass
                #----------------------------------------------
                if isinstance(payload, payloadtypes.RESTReqBase):
                    logger.debug('Handling REST request from {}'.format(payload.host))
                    try:
                        content = self.handle_rest_request(payload)
                        response = { 'status': 'success', 'content': content}
                    except Exception as e:
                        logger.warning('REST Request Exception Traceback:', exc_info=True)
                        logger.warning('Failed to fulfill REST request for {}:\n{}'.format(payload.host, str(e)))
                        response = { 'status': 'failure', 'message': str(e), 'content': None}
                    socketio.send_response(sock, response)
                #----------------------------------------------
                elif isinstance(payload, payloadtypes.SimReport):
                    logtask(logger.info, 'Received simulation report', payload)
                    self.handle_simreport(payload)
                #----------------------------------------------
                elif isinstance(payload, payloadtypes.SummaryRequest):
                    logger.debug('Sending database summary to {}'.format(payload.get('host', 'unknown')))
                    response = {'status': 'success', 'content': database.get_summary()}
                    socketio.send_response(sock, response)
                #----------------------------------------------
                elif isinstance(payload, dict) and payload['type'] == MESSAGETYPE.STATUSREQUEST:
                    logger.debug('Sending server status to {}'.format(payload.get('host', 'unknown')))
                    socketio.send_response(sock, {
                        'num_failed': database.get_number_of_docs_by_status(DBCOLLECTIONS.SIMULATION, PROCSTATUS.FAILED),
                        'num_skipped': database.get_number_of_docs_by_status(DBCOLLECTIONS.SIMULATION, PROCSTATUS.SKIPPED),
                        'num_finished': database.get_number_of_docs_by_status(DBCOLLECTIONS.SIMULATION, PROCSTATUS.FINISHED),
                        'num_pending': database.get_number_of_docs_by_status(DBCOLLECTIONS.SIMULATION, PROCSTATUS.PENDING),
                        'num_inprogress': database.get_number_of_docs_by_status(DBCOLLECTIONS.SIMULATION, PROCSTATUS.INPROGRESS),
                        'num_queued': database.get_number_of_docs_by_status(DBCOLLECTIONS.SIMULATION, PROCSTATUS.QUEUED),
                    })
                #----------------------------------------------
                elif isinstance(payload, payloadtypes.PlanStatusRequest):
                    try:
                        content = self.handle_plan_status_request(payload)
                        jsoncontent = socketio.make_json_friendly(content)
                        response = { 'status': 'success', 'content': jsoncontent}
                    except Exception as e:
                        logger.warning('Plan Status Request Exception Traceback:', exc_info=True)
                        logger.warning('Failed to fulfill plan status request for {}:\n{}'.format(payload.host, str(e)))
                        response = { 'status': 'failure', 'message': str(e), 'content': None}
                    socketio.send_response(sock, response)
                #----------------------------------------------
                else:
                    logger.warning("Unknown request type received \"{}\"".format(payload['type']))

            except Exception as e:
                logger.exception("Error while trying to handle socket connection")
            finally:
                q_connections.task_done()

#======================================================================================================
servsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
num_connection_threads = 10
q_connections = Queue(10)
connection_threads = []
num_geometry_workers = min(8, multiprocessing.cpu_count()-2)
geometry_workers = []
num_simulationscheduler_workers = 3
simulationscheduler_workers = []

def sighandler_cleanup(signum, frame):
    logger.warning('Received signal {}'.format(signum))
    signal.signal(signal.SIGINT, signal.default_int_handler)
    cleanup()

def cleanup():
    """Exit gracefully"""
    logger.info('Cleaning up before exiting')
    servsock.close()

    # allow ConnectionHandlers to finish
    for t in connection_threads:
        t.stop()
        q_connections.put(None)
    q_connections.join()
    for t in connection_threads:
        t.join()

    if not args.nocompute:
        # allow GeometryWorkers to finish
        for p in geometry_workers:
            p.stop()
            GeometryGenerator.q_geometry.put(None)
        GeometryGenerator.q_geometry.join()
        for p in geometry_workers:
            p.join()

        for p in simulationscheduler_workers:
            p.stop()
            p.join()

    logger.info("Shutting down server...")
    sys.exit(0)

if __name__ == '__main__':
    if args.test:
        # setup test environment
        # copy part of dataset
        datadir = args.data + '_test'
        dbname = args.dbname + '_test'
        shutil.rmtree(datadir)
        os.makedirs(datadir, exist_ok=True)

        database.init_dbclient(host=args.dbhost, port=args.dbport,
                                   dbname=dbname, auth=args.dbauth)
        database.InitDataStorage(datadir)

        # clear test database
        if dbname in database.dbclient.list_database_names():
            database.dbclient.drop_database(dbname)
        database.db = database.dbclient.get_database(dbname)
    else:
        # update default mongo address
        database.init_dbclient(host=args.dbhost, port=args.dbport,
                               dbname=args.dbname, auth=args.dbauth)
        database.InitDataStorage(args.data)

    if args.cleandb:
        database.cleandb_reset_corrupt_sims(dryrun=args.dryrun)
        database.cleandb_remove_leftover_files(dryrun=args.dryrun)
        sys.exit(0)

    if args.regen_gps:
        database.regenerate_gps_files()
        sys.exit(0)

    database.update_database()

    # register signal handlers for graceful shutdown
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, sighandler_cleanup)

    # create pool of socket connection handling threads
    for tidx in range(num_connection_threads):
        t = ConnectionHandler(name="Connection Handler ({})".format(tidx))
        connection_threads.append(t)
        t.start()

    # create pool of geometry generating threads
    if not args.nocompute:
        for tidx in range(num_geometry_workers):
            p = GeometryGenerator(name="Geometry Generator ({})".format(tidx), workernum=tidx)
            geometry_workers.append(p)
            p.start()

        # create a single simulation scheduler thread
        for tidx in range(num_simulationscheduler_workers):
            p = SimulationScheduler(name="Simulation Scheduler ({})".format(tidx), workernum=tidx, compute_hosts=args.computehosts)
            simulationscheduler_workers.append(p)
            p.start()

    # begin listening for connections
    servsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    servsock.bind(('', args.bindport))
    servsock.listen(5)
    logger.info('client listening for task completion on port {}'.format(args.bindport))

    ii = -1
    while True:
        ii+=1
        (clientsock, address) = servsock.accept()
        q_connections.put(clientsock)
