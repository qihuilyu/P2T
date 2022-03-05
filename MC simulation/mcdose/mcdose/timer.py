from time import perf_counter
from datetime import timedelta
class Timer():
    def __init__(self):
        self.time_start = perf_counter()
        self.time_stop = self.time_start
        self.running = False

    def start(self, description=None):
        self.description = description
        self.time_start = perf_counter()
        self.running = True

    def stop(self):
        self.time_stop = perf_counter()
        self.running = False
        return self.elapsed

    def stop_str(self):
        self.stop()
        return str(self)

    def restart(self, *args, **kwargs):
        elapsed = self.stop()
        self.start(*args, **kwargs)
        return elapsed

    def restart_str(self, *args, **kwargs):
        self.stop()
        s = str(self)
        self.start(*args, **kwargs)
        return s

    @property
    def elapsed(self):
        return timedelta(seconds=self.elapsed_seconds)

    @property
    def elapsed_seconds(self):
        return self.time_stop - self.time_start

    def __str__(self):
        return 'Timer ({desc:s}):  {elapsed!s}'.format(
                   desc=self.description if self.description is not None else 'none',
                   elapsed=self.elapsed
            )
