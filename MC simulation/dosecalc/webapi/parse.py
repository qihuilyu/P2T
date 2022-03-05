import sys, os
import argparse

import socketio
import defaults
import log
logger = log.get_module_logger(__name__)

class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        """Redefinition to print full help message on any errors"""
        self.print_help(sys.stderr)
        self.exit(2, '%s: error: %s\n' % (self.prog, message))

class DNSLookupAction(argparse.Action):
    """perform DNS query for each host passed to arg"""
    def __call__(self, parser, args, values, option_string=None):
        hosts = []
        for v in values:
            hosts += socketio.get_hosts_by_dns(v)
        setattr(args, self.dest, hosts)

def register_computeaddress_args(parser):
    parser.add_argument('--computehosts', '--ch', type=str, nargs='+',
                        default=defaults._cs_hosts,
                        help='list of compute server addresses')
    parser.add_argument('--computeport', '--cp', type=int,
                        default=defaults._cs_port,
                        help='Compute server port')

def register_db_args(parser):
    parser.add_argument('--dbhost', type=str, help='define the host for mongodb connections',
                        default=defaults.db_address[0])
    parser.add_argument('--dbport', type=int, help='define the port for mongodb connections',
                        default=defaults.db_address[1])
    parser.add_argument('--dbname', type=str, help='define the database to use',
                        default=defaults.db_name)
    parser.add_argument('--dbauth', type=str, nargs=2, metavar=('user', 'pass'), help='database authentication',
                        default=defaults.db_auth)
