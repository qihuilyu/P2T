#!/bin/bash
set -e

# activate the geant4 environment then run the command in arguments
source /usr/local/geant4/bin/geant4.sh
exec "$@"
