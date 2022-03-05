#!/bin/bash
set -eou pipefail

function echoerr() {
  printf '\e[31m%s\e[0m\n' "$@" 1>&2
}
function echowarn() {
  printf '\e[33m%s\e[0m\n' "$@" 1>&2
}
function print_usage() {
  echo "usage:  $(basename $0) [--docker-args \"<args>\"] [test-args]"
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
EXEC_PARAMS=""
while (( "$#" )); do
  case "$1" in
    --docker-args)
      DOCKER_PARAMS="$DOCKER_PARAMS $2"
      shift 2
      ;;
    --display)
      DOCKER_PARAMS="$DOCKER_PARAMS -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix"
      shift
      ;;
    --cpu)
      DOCKER_PARAMS="$DOCKER_PARAMS -e CUDA_VISIBLE_DEVICES=\"\""
      shift
      ;;
    --usage)
      print_usage
      exit 0
      ;;
    *)
      EXEC_PARAMS="$EXEC_PARAMS $1"
      shift
      ;;
  esac
done

if [[ -n EXEC_PARAMS ]]; then
  EXEC_PARAMS='discover'
fi

echo "Docker Args:  $DOCKER_PARAMS"
echo "Test Args:    $EXEC_PARAMS"
echo "Spinning up Docker Container..."

sudo docker run --rm -it \
    $(get_nvidia_runtime_args) \
    --name="mcdose-unittest" \
    -u $(id -u) \
    -e TZ="$(timedatectl status | grep 'Time zone' | awk '{print $3}')" \
    -w "/src/test" \
    -v "$(dirname $(realpath $0))/test:/src/test" \
    -v "$(realpath ./mcdose):/src/mcdose" \
    --entrypoint python \
    $DOCKER_PARAMS \
    mcdose \
    -m unittest $EXEC_PARAMS
