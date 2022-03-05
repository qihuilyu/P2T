import os

from utils import split_string

# dosecalc binary name
dosecalc_binary = 'dosecalc'

# database settings
db_address = (os.getenv('DB_HOST', default='localhost'),
              os.getenv('DB_PORT', default=27017))
db_name = os.getenv('DB_NAME', default='data')
db_auth = split_string(os.getenv("DB_AUTH", default=''))

# dataserver settings
ds_address = (os.getenv('DS_HOST', default='localhost'),
              os.getenv('DS_PORT', default=5567))

# when using docker swarm ingress network, any valid host will forward the request to the docker load balancer which will forward to a swarm node of its choosing
cs_address = []
_cs_hosts = split_string(os.getenv('CS_HOST', default='localhost'))
_cs_port = os.getenv('CS_PORT', default=5566)
for host in _cs_hosts:
    cs_address.append((host, _cs_port))
