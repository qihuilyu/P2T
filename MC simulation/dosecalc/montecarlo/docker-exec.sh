#!/bin/bash
set -eou pipefail

function print_usage() {
  echo "usage:  $(basename $0) [--docker-args \"<args>\"] [--data <workdir>] [mc args ...]"
}


DOCKER_PARAMS=""
EXEC_PARAMS=""
while (( "$#" )); do
  case "$1" in
    --docker-args)
      DOCKER_PARAMS="$DOCKER_PARAMS $2"
      shift 2
      ;;
    --data)
      OUTPUT_DIR="$2"
      shift 2
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

OUTPUT_DIR="$(realpath ${OUTPUT_DIR:=.})"

echo "Docker Args:  $DOCKER_PARAMS"
echo "Spinning up Docker Container..."

mkdir -p "${OUTPUT_DIR}/results"
sudo docker run --rm -it \
    --name="clouddose-mcsim" \
    -u $(id -u) \
    -e TZ="$(timedatectl status | grep 'Time zone' | awk '{print $3}')" \
    -w "/data" \
    -v "${OUTPUT_DIR}:/data" \
    --entrypoint "/entry-point.sh" \
    $DOCKER_PARAMS \
    clouddose-compute "/usr/local/geant4/applications/bin/dosecalc" --individual --outputdir /data/results ${EXEC_PARAMS}

