from datetime import datetime, timedelta
import threading
import multiprocessing

class WorkerStopped(Exception):
    """raised when a StoppableThread is stopped via its stop() method
    This exception allows the threads run() method to handle stop requests
    uniformly no matter when it occurs"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ScheduledTask():
    """Runs a task defined by the function assigned to task_cb at an interval specified by the
    datetime.timedelta object in frequency

    The managing thread must call update() periodically or the task will not be executed
    """
    def __init__(self, task_cb, frequency):
        self.task_cb   = task_cb
        self.frequency = frequency
        self.last_call_time = datetime.now()
        self.first_time = True
        assert isinstance(self.frequency, timedelta)

    def update(self):
        now = datetime.now()
        if self.first_time or (now - self.last_call_time) >= self.frequency:
            self.first_time = False
            self.last_call_time = now
            self.task_cb()

class StoppableThread(threading.Thread):
    """thread class with stop() method that can interrupt system calls"""
    def __init__(self, *args, workernum=-1, **kwargs):
        super().__init__(**kwargs)
        self._stop_event = threading.Event()
        self._idle_event = threading.Event()
        self.workernum=workernum

    def stop(self):
        """signal graceful exit of infinite thread"""
        self._stop_event.set()

    def _setidle(self, isidle=True):
        """should be updated by thread during run() method to keep is_idle() up to date for callers"""
        if isidle:
            self._idle_event.set()
        else:
            self._idle_event.clear()

    def is_idle(self):
        """get idle status. relies on thread updating _idle_event flag"""
        return self._idle_event.is_set()

    def is_stopped(self):
        """get stopped status"""
        return self._stop_event.is_set()
