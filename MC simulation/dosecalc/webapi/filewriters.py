import os
import log

import numpy as np

logger = log.get_module_logger(__name__)

class RotatingNPFileWriter():
    def __init__(self, fname, max_fsize=4, texthead=None, *args, **kwargs):
        """Opens one or more files with name <fname>_###.npy, caching samples into np array until max_fsize is reached
        Then the npy file is saved and a new file is started

        Args:
            fname (str): base filename
            max_fsize (float): max filesize before rotating (in GB)
        """
        self.fnamebase = os.path.splitext(fname)[0]
        self.fileid = 0
        self.max_fsize = max_fsize*(1024**3)
        self.cur_fsize = 0
        self.arrshape = None
        self.databuf = []
        self.texthead = texthead
        self.textbuf = []

    def __del__(self):
        if len(self.databuf):
            self.save()

    def filename(self, ext='.npy'):
        return '{}_{:03d}{}'.format(self.fnamebase, self.fileid, ext)

    def rotate(self):
        """increment filename, reset buffer, and return next usable filename"""
        self.fileid += 1
        self.cur_fsize = 0
        self.databuf = []
        self.textbuf = []
        return self.filename('.npy')

    def save(self):
        """save contents of databuf to the current file"""
        if len(self.databuf):
            arr = np.concatenate(self.databuf, axis=0)
            arr = np.ascontiguousarray(arr)
            fname = self.filename('.npy')
            np.save(fname, arr)
            logger.info('Saved data with shape {!s} to file: "{}"'.format(arr.shape, fname))
            if len(self.textbuf):
                with open(self.filename('.txt'), 'w') as fd:
                    fd.write(self.texthead+'\n')
                    for line in '\n'.join([str(x) for x in self.textbuf]):
                        fd.write(line)
        self.rotate()

    def write(self, arr, text=[]):
        """push new array to buffer, possibly write file and rotate if max_fsize is exceeded"""
        assert isinstance(arr, np.ndarray)
        fsize = arr.nbytes
        if self.cur_fsize + fsize >= self.max_fsize:
            self.save()

        if self.arrshape is None:
            self.arrshape = arr.shape
        assert arr.shape[1:] == self.arrshape[1:]
        self.cur_fsize += fsize
        self.databuf.append(arr)
        if not isinstance(text, list):
            text = [text]
        self.textbuf += text
