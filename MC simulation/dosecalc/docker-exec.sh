#!/bin/bash
set -euo pipefail

function print_usage() {
  echo "usage:  $(basename $0) --dbdata <db-data-path> --results <result-path> -- <python-file> [args-to-python-file]"
}
function get_nvidia_runtime_args() {
  # emit the correct nvidia-docker command line arguments based on the docker version
  if sudo docker run --help | grep -e '--gpus' 2>&1 >/dev/null; then
    echo '--gpus=all'
  else
    echo '--runtime=nvidia'
  fi
}

DOCKER_PARAMS=""
PARAMS=""
IGNORED=""
while (( "$#" )); do
  case "$1" in
    --dbdata)
      DBDATA="$2"
      shift 2
      ;;
    --results|--result|--out|-o)
      OUT="$2"
      shift 2
      ;;
    --shell)
      SH=1
      shift
      ;;
    --nomount)
      NOMOUNT=1
      shift
      ;;
    --docker-args)
      DOCKER_PARAMS="$DOCKER_PARAMS $2"
      shift 2
      ;;
    --) # remaining args should be passed as is
      shift
      PARAMS="$PARAMS $@"
      break
      ;;
    -h|--help|-u|--usage)
      print_usage
      exit 0
      ;;
    *) 
      IGNORED="$IGNORED $1"
      shift
      ;;
  esac
done
PARAMS=($PARAMS) # convert to array
PYFILE="${PARAMS[@]:0:1}"
PARAMS=${PARAMS[@]:1}

if [[ ${NOMOUNT:=0} = 1 ]]; then
  vol_mounts=""
else
  vol_mounts="\
  -v $(dirname $(realpath $0))/webapi:/src/webapi \
  -v $(dirname $(realpath $0))/../mcdose/mcdose:/src/mcdose"
fi

# set entrypoint and associated args
if [[ ${SH:-0} = 1 ]]; then
  entrypoint="bash"
  entry_args=""
  OUT=${OUT:=$(realpath $(pwd))}
  DBDATA=${DBDATA=$(realpath $(pwd))}
else
  # argument verification
  if ! [[ -v OUT && -v DBDATA && -n "$PYFILE" ]]; then
    print_usage
    exit 1
  fi

  entrypoint='python'
  entry_args="/src/webapi/$PYFILE $PARAMS"
fi

echo "Docker Args:   $DOCKER_PARAMS"
echo "Exec Args:     $entry_args"
echo "Ignored args:  $IGNORED"
echo "Spinning up Docker container..."

sudo docker run --rm \
  $(get_nvidia_runtime_args) \
  -it --network host \
  -u $(id -u) \
  -e TZ="$(timedatectl status | grep 'Time zone' | awk '{print $3}')" \
  -v "${DBDATA}:/dbdata" \
  -v "${OUT}:/results" \
  ${vol_mounts} \
  -e PYTHONPATH="/src" \
  --workdir "/result" \
  --entrypoint "${entrypoint}" \
  $DOCKER_PARAMS \
  clouddose-client ${entry_args}
