#!/bin/bash
set -eou pipefail

function echoerr() {
  printf '\e[31m%s\e[0m\n' "$@" 1>&2
}
function echowarn() {
  printf '\e[33m%s\e[0m\n' "$@" 1>&2
}
function print_usage() {
  echo "usage:  $(basename $0) [--docker-args \"<args>\"] [--exec-args \"<args>\"] --data <data-dir> --action {train,test,predict} [action-args]"
}
function get_nvidia_runtime_args() {
  # emit the correct nvidia-docker command line arguments based on the docker version
  if sudo docker run --help | grep -e '--gpus' 2>&1 >/dev/null; then
    echo '--gpus=all'
  else
    echo '--runtime=nvidia'
  fi
}

DATAVOL=
ACTION=
DOCKER_PARAMS=""
EXEC_PARAMS=""
ACTION_PARAMS=""
datadir="traindata"
rundir="runs"
while (( "$#" )); do
  case "$1" in
    -d|--data)
      DATA="$2"
      shift 2
      ;;
    -a|--action)
      ACTION="$2"
      shift 2
      ;;
    --docker-args)
      DOCKER_PARAMS="$DOCKER_PARAMS $2"
      shift 2
      ;;
    --display)
      DOCKER_PARAMS="$DOCKER_PARAMS -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix"
      shift
      ;;
    --exec-args)
      EXEC_PARAMS="$2"
      shift 2
      ;;
    --datadir)
      datadir="$2"
      shift 2
      ;;
    --rundir)
      rundir="$2"
      shift 2
      ;;
    --usage)
      print_usage
      exit 0
      ;;
    *)
      ACTION_PARAMS="$ACTION_PARAMS $1"
      shift
      ;;
  esac
done

case "$ACTION" in
  train|test)
    ACTION_PARAMS="--rundir /data/${rundir} --datadir /data/${datadir} $ACTION_PARAMS"
    ;;
  predict)
    ;;
  *)
    echoerr "Action must be one of {train, test, predict}, not \"${ACTION}\""
    print_usage
    exit 1
esac

if [[ ! -n ${DATA-} ]]; then
  echoerr '"--data <data-dir>" must be specified'
  print_usage
  exit 1
fi

echo "Action:       $ACTION"
echo "Docker Args:  $DOCKER_PARAMS"
echo "Exec Args:    $EXEC_PARAMS"
echo "Action Args:  $ACTION_PARAMS"
echo "Spinning up Docker Container..."

sudo docker run --rm -it \
    $(get_nvidia_runtime_args) \
    -u $(id -u) \
    -e TZ="$(timedatectl status | grep 'Time zone' | awk '{print $3}')" \
    -w "/src" \
    -v "$(realpath ./mcdose):/src/mcdose" \
    -v "${DATA}:/data" \
    $DOCKER_PARAMS \
    mcdose $EXEC_PARAMS \
    "$ACTION" $ACTION_PARAMS
