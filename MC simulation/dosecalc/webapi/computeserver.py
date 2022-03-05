
import sys, os
from os.path import join as pjoin
import shutil
import socket
import signal
import queue
import subprocess
import uuid
import time
import argparse
import log

import defaults
import payloadtypes
from workertypes import StoppableThread, WorkerStopped
from api_enums import (MESSAGETYPE, STATUS)
import socketio
from generate_input import generate_init, generate_beamon

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', type=str, default="compute_data", help="set data root directory")
parser.add_argument('--noclean', action='store_true', help="don't cleanup temporary data after task completion")
parser.add_argument('--bindport', type=int, help='port on which this service is listening',
                    default=defaults._cs_port)
log.add_argument_loglevel(parser)
args = parser.parse_args()

logger = log.get_module_logger(__name__, level=args.loglevel)

curdir = os.path.abspath(os.path.dirname(__file__))
DATA         = args.data
TEMP         = pjoin(DATA, "temp")
f_executable = defaults.dosecalc_binary # added to PATH in docker image
f_init       = "init.in"
f_beamon     = "beamon.in"

class SimulationTask():
    def __init__(self):
        self.reply_host = None
        self.reply_port = None
        self.payload    = None

class ReportTask():
    def __init__(self):
        self.simtask    = None
        self.workdir    = None
        self.payload    = None

class SimulationWorker(StoppableThread):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def stop(self):
        logger.debug('Stopping thread "{}"'.format(self.getName()))
        super().stop()

    @staticmethod
    def init_task(workdir, num_vacant_threads=0, magfield=(0, 0, 1.5, 'tesla')):
        """place static MC sim files into workdir"""
        with open(pjoin(workdir, f_init), 'w') as fd:
            generate_init(fd, nthreads=os.cpu_count()-num_vacant_threads,
                          magfield=magfield)

    @staticmethod
    def make_temp_directory(id, root=TEMP):
        """create temporary working directory under root with random name"""
        while True:
            try:
                dname = pjoin(TEMP, '{}_{}'.format(id, uuid.uuid4().hex[:8]))
                os.makedirs(dname)
                break
            except FileExistsError as e: pass
        return dname

    def _run_sim(self, resultdir, sim, geometry_file, gps_file, addl_args=[], num_vacant_threads=0):
        os.makedirs(resultdir, exist_ok=True)
        with open(pjoin(resultdir, f_beamon), 'w') as fd:
            generate_beamon(fd, int(sim.num_particles), int(sim.num_runs))

        # generate init file
        self.init_task(resultdir, num_vacant_threads=num_vacant_threads,
                       magfield=sim.magnetic_field)

        # generate call args
        if addl_args is None:
            addl_args = []
        addl_args = addl_args if isinstance(addl_args, (list, tuple)) else [addl_args]
        call_args = [
            f_executable,
            *addl_args,
            '-i',
            pjoin(os.path.pardir, geometry_file),
            pjoin(f_init),
            pjoin(os.path.pardir, gps_file),
            pjoin(f_beamon),
        ]
        try:
            logfd = open(pjoin(resultdir, 'run_log.txt'), 'w')
            logfd.write('args: "'+', '.join(str(x) for x in call_args)+'"\n\n')
            logfd.flush()
            proc = subprocess.Popen(call_args, cwd=resultdir, stdout=logfd, stderr=subprocess.STDOUT, encoding='utf-8')
            while proc.poll() is None:
                if self.is_stopped():
                    logger.debug('Terminating simulation subprocess')
                    proc.terminate()
                    raise WorkerStopped()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, proc.args)
        except (WorkerStopped, subprocess.CalledProcessError, subprocess.SubprocessError) as err:
            if isinstance(err, WorkerStopped):
                err_string = "\nException: Simulation subprocess was terminated early due to an unexpected application shutdown"
            else:
                logger.warning("Error during MC simulation subprocess:\n{}".format(str(err)))
                if isinstance(err, subprocess.CalledProcessError):
                    err_string = '\nReturn code: {:d}\nException:\n{}'.format(proc.returncode, str(err))
                else:
                    err_string = '\nUnknwon error occured\nException:\n{}'.format(str(err))
            logfd.write(err_string)
            raise
        finally:
            logfd.close()

    def _run_task(self, task: SimulationTask):
        taskid = task.payload.id
        result = payloadtypes.SimReport()
        result.status = STATUS.FAILURE
        result.message = None
        result.beam_id = task.payload.beam_id
        result.subbeam_id = task.payload.subbeam_id
        result.simulations = [
            payloadtypes.SimulationResult.fromdict({'id': sim.id, 'files': {}, '_resultdir': None, 'time_elapsed': None, 'host': socketio.gethostname()})
            for sim in task.payload.simulations
        ]
        workdir = None
        try:
            if self.is_stopped():
                # short circuit to finish remaining queued tasks with "failure status"
                raise WorkerStopped()
            workdir = self.make_temp_directory(task.payload.subbeam_id)
            logger.debug('Preparing simulation ({})'.format(workdir))
            for label, files in task.payload.files.items():
                socketio.unpack_files(workdir, files)
            logger.debug('Running simulation ({})'.format(workdir))
            for sim in task.payload.simulations:
                simid = sim.id
                resultdir = pjoin(workdir, simid)
                os.makedirs(resultdir)
                time_start = time.perf_counter()
                self._run_sim(resultdir, sim, task.payload.files['geometry']['name'], task.payload.files['gps']['name'], addl_args=sim.callargs)
                resultsim = next(sim for sim in result.simulations if sim.id == simid)
                resultsim.time_elapsed = time.perf_counter()-time_start
                resultsim._resultdir = resultdir # temporary for use in ReportWorker
        except Exception as e:
            raise
            # cleanup unfinished simulation and queue up a notification
            if isinstance(e, WorkerStopped):
                logger.warning('Task was stopped early ({})'.format(taskid))
                result.message = "Task was stopped early by signal"
            else:
                logger.warning('Problem while running task ({}):\n'.format(taskid, str(e)))
                result.message = str(e)
            result.status = STATUS.FAILURE
        else:
            logger.debug('Completed task ({})'.format(taskid))
            result.status = STATUS.SUCCESS
        finally:
            # queue task confirmation
            logger.debug('Queuing status notification ({})'.format(taskid))
            reporttask = ReportTask()
            reporttask.simtask = task
            reporttask.workdir = workdir
            reporttask.payload = result
            q_reporttasks.put(reporttask)

    def run(self):
        """dequeue task and unpack payload to prepare for MC simulation"""
        # infinite loop until thread is stopped
        while True:
            taskid = 'unknown'
            try:
                self._setidle()
                task = q_pendingtasks.get()
                if task is None:
                    logger.debug('exiting thread {}'.format(self.getName()))
                    break
                assert(isinstance(task, SimulationTask))
                assert(isinstance(task.payload, payloadtypes.SimInstruction))
                for sim in task.payload.simulations:
                    assert(isinstance(sim, payloadtypes.SimulationConfig))
                taskid = task.payload.id
                logger.info("Processing task ({})".format(taskid))
                self._setidle(False)
                self._run_task(task)
            except AssertionError as e:
                logger.error('Simulation task does not match required format')
            except Exception as e:
                # protect thread in case anything slips through
                logger.exception("Severe error has occured. Task has been lost ({})".format(taskid))
            finally:
                q_pendingtasks.task_done()


class ReportWorker(StoppableThread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def stop(self):
        logger.debug('Stopping thread "{}"'.format(self.getName()))
        super().stop()

    @staticmethod
    def cleanup_task_data(workdir):
        shutil.rmtree(workdir)

    def _run_report(self, task: ReportTask):
        try:
            taskid = task.workdir
            logger.debug('Sending simulation results for task ({})'.format(taskid))
            for sim in task.payload.simulations:
                filebufs = {}
                # only package files in resultdir
                root = sim._resultdir
                if root is not None:
                    # get number of runs
                    runs = [d for d in os.listdir(root) if os.path.isdir(pjoin(root, d))]
                    for run in runs:
                        rundir = pjoin(root, run)
                        bufs = []
                        runfiles = [f for f in os.listdir(rundir) if os.path.isfile(pjoin(rundir, f))]
                        for f in runfiles:
                            bufs.append(socketio.pack_file_binary(pjoin(rundir, f)))
                        filebufs[run] = bufs
                    filebufs[None] = []
                    for f in [f for f in os.listdir(root) if os.path.isfile(pjoin(root, f))]:
                        filebufs[None].append(socketio.pack_file_binary(pjoin(root, f)))
                    sim.files = filebufs
        except Exception as e:
            task.payload.status = STATUS.FAILURE
            task.payload.message += '\nError while packing result files for transfer\nException:\n{}'.format(str(e))
            raise
        finally:
            time_start = time.perf_counter()
            while True:
                try:
                    if self.is_stopped() and time.perf_counter()-time_start > 10:
                        raise TimeoutError('Timeout while trying to send task status')
                    response = task.payload.send((task.simtask.reply_host, task.simtask.reply_port), timeout=None, connection_timeout=1)
                    break
                except (OSError, ConnectionError) as e:
                    # all socket exceptions are subclassed from OSError
                    logger.warning('ConnectionError while sending results to dataserver')
                    if time.perf_counter()-time_start > 30:
                        raise socketio.TimeoutError('Timeout while trying to send task "{!s}" results to data server ({!s})'.format(
                            taskid, task.simtask.reply_host, task.simtask.reply_port))
                    time.sleep(0.1)
                    logger.warning('Trying again')
            if not args.noclean and task.workdir is not None:
                logger.debug('Cleaning up temporary data for task ({})'.format(taskid))
                self.cleanup_task_data(task.workdir)

    def run(self):
        #infinite loop
        taskid = 'unknown'
        while True:
            try:
                # wait until a task is finished
                self._setidle()
                task = q_reporttasks.get()
                if task is None:
                    logger.debug('exiting thread {}'.format(self.getName()))
                    break
                assert(isinstance(task, ReportTask))
                assert(isinstance(task.simtask, SimulationTask))
                assert(isinstance(task.payload, payloadtypes.SimReport))
                self._setidle(False)
                self._run_report(task)
                taskid = task.simtask.payload.id
                logger.info("Successfully completed task ({})".format(taskid))
            except AssertionError as e:
                logger.error('Reporting task does not match required format')
            except Exception as e:
                # protect thread in case anything slips through
                logger.exception("Severe error has occured while sending status notification. Task has been lost ({}): {}".format(taskid, str(e)))
            finally:
                q_reporttasks.task_done()

######################################################
servsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
simthread = SimulationWorker(name="Simulation Worker")
numreport_threads = 3
report_threads = []
q_pendingtasks = queue.Queue(2)
q_reporttasks = queue.Queue(10)

def sighandler_cleanup(signum, frame):
    logger.info('Received signal {}'.format(signum))
    signal.signal(signal.SIGINT, signal.default_int_handler)
    cleanup()

def cleanup():
    """Exit gracefully"""
    logger.info('Cleaning up before exiting')
    servsock.close()

    # all remaining tasks are considered failed and notifications are queued so nothing is left behind
    simthread.stop()
    q_pendingtasks.join()
    q_pendingtasks.put(None) # prompts threads to end
    simthread.join()

    # allow ReportWorkers to finish
    for t in report_threads:
        t.stop()
    q_reporttasks.join()
    for _ in report_threads:
        q_reporttasks.put(None)
    for t in report_threads:
        t.join()

    logger.info("Shutting down server...")
    sys.exit(0)


if __name__ == "__main__":
    # register signal handlers for graceful shutdown
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, sighandler_cleanup)

    try:
        # initialize directories
        shutil.rmtree(TEMP, ignore_errors=True)
        for d in [TEMP]:
            os.makedirs(d, exist_ok=True)

        # start task handling worker thread
        simthread.start()

        # start a pool of task confirm threads
        for tidx in range(3):
            t = ReportWorker(name="Report Worker ({})".format(tidx))
            report_threads.append(t)
            t.start()

        # open a listening server socket
        servsock.bind(('', args.bindport))
        servsock.listen(5) # make a server port
        logger.info('Server listening on port {}'.format(args.bindport))

        while True:
            # block until client connection
            logger.debug('Waiting for new task requests...')
            (clientsock, address) = servsock.accept()
            try:
                logger.debug('Incoming connection from address "{!s}"'.format(address))
                payload = socketio.receive_all(clientsock)
            except Exception as e:
                logger.warning('communication with requestor failed')
                continue
            try:
                payload = payloadtypes.BasePayload.fromdict(payload)
            except TypeError as e:
                pass
            #-----------------------------------------------------------------
            if isinstance(payload, payloadtypes.SimInstruction):
                if q_pendingtasks.full():
                    logger.debug('Simulation task queue is full. Not accepting new task requests until space is available.')
                    socketio.send_response(clientsock, {'status': STATUS.FAILURE})
                    # wait for space on queue
                    time.sleep(1)
                    continue
                else:
                    simtask = SimulationTask()
                    simtask.reply_host = payload.reply_host if payload.reply_host else address(0)
                    simtask.reply_port = payload.reply_port if payload.reply_host else defaults.ds_address[1]
                    simtask.payload = payload
                    q_pendingtasks.put(simtask)
                    socketio.send_response(clientsock, {'status': STATUS.SUCCESS})
                    logger.info('Scheduled task "{}"'.format(payload.id))
            #-----------------------------------------------------------------
            elif payload['type'] == MESSAGETYPE.STATUSREQUEST:
                logger.debug('Sending server status to {}'.format(address[0]))
                running_notifications = 0
                for t in report_threads:
                    if not t.is_idle():
                        running_notifications += 1
                socketio.send_response(clientsock, {
                    'running_simulations': 0 if simthread.is_idle() else 1,
                    'running_reports': running_notifications,
                    'queued_simulations': q_pendingtasks.qsize(),
                    'queue_confirmations': q_reporttasks.qsize()
                })
            #-----------------------------------------------------------------

    except Exception as e:
        logger.exception('A fatal error has occured causing the application to shutdown')
        cleanup()
