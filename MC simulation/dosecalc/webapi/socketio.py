import sys, os
from os.path import join as pjoin
import socket
import json
import base64
from datetime import datetime
from bson.objectid import ObjectId
import log
import copy

import pathvalidate

logger = log.get_module_logger(__name__)

CHUNKSIZE = 1024*1024*4 # 8MB at a time

default_receipt_timeout=20
default_recv_timeout=20

# Intended as the first byte read from every tcp packet to steer the rest of the socket.recv() process
class PAYLOAD():
    @classmethod
    def alltypes(cls):
        return [t for (name, t) in cls.__dict__.items() if not name.startswith('__')]

    #used for a single round of one-way communication
    JSON     = b'\x01'
    BINARY   = b'\x02'

    # used for a single round of two-way communication
    REQUEST  = b'\x04'
    RESPONSE = b'\x05'

class RECEIPT():
    SUCCESS = b'\xA1'
    FAILURE = b'\xA2'

class TimeoutError(socket.timeout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ReceiptError(socket.timeout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

#####################
# PRIVATE METHODS
#####################
def gethostname():
    """try to get docker host's hostname rather than container hostname first"""
    try:
        with open('/etc/host_hostname', 'r') as fd:
            return fd.read().strip('\n ')
    except Exception as e:
        return socket.gethostname()

def get_hostname_by_addr(addr):
    """return hostname for host at address"""
    try:
        (hostname, aliaslist, ipaddrlist) = socket.gethostbyaddr(
            socket.gethostbyname(addr)
        )
        return hostname
    except:
        return addr

def recv_payload_size(sock, timeout=None):
    return int.from_bytes(recv_all_chunks(sock, 4, timeout), sys.byteorder)

def recv_all_chunks(sock, payloadsize, timeout=None):
    """read a bytearray payload from tcp socket"""
    current_timeout = sock.gettimeout()
    if current_timeout != timeout:
        sock.settimeout(timeout)
    recvbuf = bytearray()
    mark = 0
    while mark < payloadsize:
        recv_size = min(CHUNKSIZE, payloadsize-mark)
        msg = sock.recv(recv_size)
        if not len(msg):
            raise BrokenPipeError('Error while receiving socket data')
        recvbuf.extend(msg)
        mark += len(msg)
    if current_timeout != timeout:
        sock.settimeout(current_timeout)
    return recvbuf

def graceful_shutdown(sock, shuttype=None):
    """close socket after waiting for shutdown from other end"""
    if shuttype is not None:
        sock.shutdown(shuttype)
    #  while True:
    #      if not sock.recv(1):
    #          break
    sock.close()

def confirm_receipt(sock, timeout):
    try:
        receipt = recv_all_chunks(sock, 1, timeout)
        if receipt == RECEIPT.SUCCESS:
            return
    except (socket.timeout, ConnectionError) as e:
        raise ReceiptError('Timeout while waiting for socket receipt')
    except (ConnectionError) as e:
        raise ReceiptError('ConnectionError while waiting for socket receipt')

def send_payload_and_confirm(sock, payload, timeout):
    """low-level send-then-recv for single message socket communication"""
    sock.sendall(payload)
    sock.shutdown(socket.SHUT_WR)
    confirm_receipt(sock, timeout)
    graceful_shutdown(sock)

def wrap_payload(payload, payloadtype):
    if payloadtype in (PAYLOAD.JSON, PAYLOAD.REQUEST, PAYLOAD.RESPONSE):
        payload['host'] = gethostname()
        payload = bytearray(json.dumps(payload), 'utf-8')
    header = payloadtype + len(payload).to_bytes(4, sys.byteorder)
    return header + payload

def null2none(doc, inplace=False):
    """replace 'null' keys and values with python None"""
    if inplace:
        newdoc = doc
    else:
        newdoc = copy.copy(doc)
    if isinstance(newdoc, dict):
        for k, v in list(newdoc.items()):
            if isinstance(k, str) and k.lower() == 'null':
                del newdoc[k]
                k = None
            newdoc[k] = null2none(v, inplace)
    elif isinstance(newdoc, (list, tuple)):
        for ii, v in enumerate(newdoc):
            newdoc[ii] = null2none(v, inplace)
    elif isinstance(newdoc, str) and newdoc.lower() == 'null':
        newdoc = None
    return newdoc

def get_hosts_by_dns(hosts):
    if isinstance(hosts, str):
        hosts = [hosts]

    ips = []
    for host in hosts:
        try:
            ips += [x for x in socket.gethostbyname_ex(host)[2]]
        except socket.gaierror as e:
            ips.append(host)
        except Exception as e:
            logger.exception(e)
    return ips

def make_json_friendly(doc):
    """convert all non-json types to json types"""
    v = doc
    if isinstance(v, (list, tuple)):
        l = []
        for item in v:
            l.append(make_json_friendly(item))
        v = l
    elif isinstance(v, dict):
        d = {}
        for k, item in v.items():
            d[k] = make_json_friendly(item)
        v = d
    elif isinstance(v, ObjectId):
        v = str(v)
    elif isinstance(v, datetime):
        v = v.strftime('%F_%T%z')
    return v


#####################
# USER FACING METHODS
#####################
def pack_numpy_array(arr, name):
    """pack numpy array into json-serializable base64 encoded string"""
    return {'name': name, 'dtype': str(arr.dtype), 'size': arr.shape, 'type': 'numpy', 'contents': str(base64.b64encode(arr.data))[2:-1]}

def pack_file_binary(file, name=None):
    """read bytes from files and pack into json as base64-encoded string"""
    if not name:
        name = os.path.basename(file)
    with open(file, 'rb') as fd:
        return {'name': name, 'type': 'binary', 'contents': str(base64.b64encode(fd.read()))[2:-1]}

def pack_file_text(file, name=None):
    """read text from file and pack into json as string"""
    if not name:
        name = os.path.basename(file)
    with open(file, 'r') as fd:
        return {'name': name, 'type': 'text', 'contents': fd.read()}

def unpack_file_to_memory(filespec):
    """unpack json serialied filebuffer into memory as string or bytes"""
    if filespec['type'] == 'text':
        return filespec['contents']
    else:
        return base64.b64decode(filespec['contents'])

def unpack_file(workdir, filespec, name=None):
    """write json serialied filebuffer into stream"""
    if not name:
        name = filespec['name']
    name = pathvalidate.sanitize_filepath(name)
    fname = pjoin(workdir, name)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    openmode = 'w' if filespec['type'] == 'text' else 'wb'
    with open(fname, openmode) as fd:
        fd.write(unpack_file_to_memory(filespec))
    logger.debug('Unpacked {!s} file: "{!s}"'.format(filespec['type'], fname))
    return fname

def unpack_files(workdir, filespecs):
    """convert file streams into files on disk"""
    fnames = []
    if not isinstance(filespecs, list):
        filespecs = [filespecs]
    for f in filespecs:
        fname = unpack_file(workdir, f)
        fnames.append(fname)
    return fnames

def send_request(sock, data, timeout=default_recv_timeout):
    """2-way communication; similar to send_payload but expects to receive data back from listener rather
    than just a receipt. TimeoutError will be raised by sender if no response is given by receiver.
    """
    payload = wrap_payload(data, PAYLOAD.REQUEST)
    sock.sendall(payload)
    response = receive_all(sock, timeout)
    graceful_shutdown(sock)
    return response

def send_response(sock, data={}):
    """send as a reply to receiving a PAYLOAD.RESPONSE query, attaching arbitrary JSON data as payload
    This implements a single 2-way communication unlike the send_payload method that only allow a one-way
    communication with an automatic and generic receipt confirmation. This response is required or else,
    the sender will raise a TimeoutError.
    """
    payload = wrap_payload(data, PAYLOAD.RESPONSE)
    sock.sendall(payload)
    graceful_shutdown(sock)

def send_payload(sock, data, timeout=default_receipt_timeout):
    """Send payload as a 1-way message with automatic reciept confirmation and no return data"""
    if isinstance(data, (bytearray, bytes)):
        """should be valid for files up to 4GB"""
        payload = wrap_payload(data, PAYLOAD.BINARY)
    else:
        payload = wrap_payload(data, PAYLOAD.JSON)
    send_payload_and_confirm(sock, payload, timeout)

def receive_all(sock, timeout=default_recv_timeout):
    """interpret message type and return appropriate data"""
    payloadtype = recv_all_chunks(sock, 1, timeout)
    if payloadtype in PAYLOAD.alltypes():
        try:
            payloadsize = recv_payload_size(sock, timeout)
            data = recv_all_chunks(sock, payloadsize, timeout)
            data = null2none(json.loads(data.decode('utf-8')), inplace=True)
            data['address'] = sock.getpeername()[0]
            if payloadtype not in (PAYLOAD.REQUEST, PAYLOAD.RESPONSE):
                sock.shutdown(socket.SHUT_RD)
                sock.sendall(RECEIPT.SUCCESS)
                graceful_shutdown(sock)
        except Exception as e:
            sock.shutdown(socket.SHUT_RD)
            sock.sendall(RECEIPT.FAILURE)
            graceful_shutdown(sock)
            raise
    else:
        raise RuntimeError('recieved unknown data payload type "{!s}"'.format(payloadtype))
    return data

