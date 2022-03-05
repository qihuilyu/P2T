# encoding: utf-8

"""
Stdout/stderr redirector, at the OS level, using file descriptors.
This also works under windows.
"""

__docformat__ = "restructuredtext en"

#-------------------------------------------------------------------------------
#  Copyright (C) 2008  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-------------------------------------------------------------------------------


import os
import sys
import io

STDOUT = 1
STDERR = 2

class FDRedirector(object):
    """ Class to redirect output (stdout or stderr) at the OS level using
        file descriptors.
    """
    def __init__(self, fd=STDOUT):
        """ fd is the file descriptor of the outpout you want to capture.
            It can be STDOUT or STERR.
        """
        self.fd = fd
        self.started = False
        self.piper = None
        self.pipew = None

    def start(self):
        """ Setup the redirection.
        """
        if not self.started:
            self.oldhandle = os.dup(self.fd)
            self.piper, self.pipew = os.pipe()
            os.dup2(self.pipew, self.fd)
            os.close(self.pipew)
            if self.fd == STDOUT:
                sys.stdout = io.TextIOWrapper(os.fdopen(self.oldhandle, 'wb'))
            elif self.fd == STDERR:
                sys.stderr = io.TextIOWrapper(os.fdopen(self.oldhandle, 'wb'))

            self.started = True

    def flush(self):
        """ Flush the captured output, similar to the flush method of any
        stream.
        """
        if self.fd == STDOUT:
            sys.stdout.flush()
        elif self.fd == STDERR:
            sys.stderr.flush()

    def stop(self):
        """ Unset the redirection and return the captured output.
        """
        if self.started:
            self.flush()
            os.dup2(self.oldhandle, self.fd)
            os.close(self.oldhandle)
            f = os.fdopen(self.piper, 'r')
            output = f.read()
            f.close()

            self.started = False
            return output
        else:
            return ''

    def getvalue(self):
        """ Return the output captured since the last getvalue, or the
        start of the redirection.
        """
        output = self.stop()
        self.start()
        return output


#  import io
#  import tempfile
#  import ctypes
#  from contextlib import contextmanager
#  libc = ctypes.CDLL(None)
#  c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
#  c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')
#  print(c_stdout)
#  print(c_stderr)
#  @contextmanager
#  def stdout_redirector(stream):
#      original_stdout_fileno = sys.stdout.fileno()
#      original_stderr_fileno = sys.stderr.fileno()

#      def _redirect_stdout(to_fd):
#          """redirect stdout to the given file descriptor"""
#          # flush the C-level buffer stdout
#          libc.fflush(c_stdout)
#          # Flush and close sys.stdout - also closes file descriptor
#          print('closing stdout')
#          sys.stdout.close()
#          # make original_stdout_fileno point to the same file as to_fd
#          os.dup2(to_fd, original_stdout_fileno)
#          # Create a new sys.stdout that points to the redirected fd
#          sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fileno, 'wb'))

#      def _redirect_stderr(to_fd):
#          """redirect stdout to the given file descriptor"""
#          # flush the C-level buffer stderr
#          libc.fflush(c_stderr)
#          # Flush and close sys.stderr - also closes file descriptor
#          print('closing stderr')
#          sys.stderr.close()
#          # make original_stderr_fileno point to the same file as to_fd
#          os.dup2(to_fd, original_stderr_fileno)
#          # Create a new sys.stdout that points to the redirected fd
#          sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fileno, 'wb'))

#      # save a copy of original stdout fd in saved_stdout_fd
#      saved_stdout_fileno = os.dup(original_stdout_fileno)
#      saved_stderr_fileno = os.dup(original_stderr_fileno)
#      try:
#          # create temporary file and redirect stdout to it
#          tfile = tempfile.TemporaryFile(mode='w+b')
#          _redirect_stdout(tfile.fileno())
#          _redirect_stderr(tfile.fileno())
#          sys.stdout.write("printing to new stdout\n")
#          # yield to caller, then redirect stdout back to the saved fd
#          yield
#          # restore original stdout/stderr fd
#          _redirect_stdout(saved_stdout_fileno)
#          _redirect_stderr(saved_stderr_fileno)
#          # copy contents of temp file to given stream
#          tfile.flush()
#          tfile.seek(0, io.SEEK_SET)
#          stream.write(tfile.read())
#      finally:
#          tfile.close()
#          os.close(saved_stdout_fileno)
#          os.close(saved_stderr_fileno)

