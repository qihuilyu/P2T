#!/bin/bash
set -eou pipefail

IMAGE="clouddose"
IMAGE_VERS="2.1.11"
REGISTRY="127.0.0.1:5000"
G4IMAGE="andreadotti/geant4"
G4VERS="10.3.p02"

function print_usage() {
  echo "$(basename $0)  [--regenerate | --usage]"
}

PARAMS=""
while (( "$#" )); do
  case "$1" in
    -r|--regen|--regenerate)
      REGEN=1
      echo "Regenerating Geant4 binary and Compute image"
      shift 1
      ;;
    -f|--force)
      FORCE=1
      REGEN=1
      echo "Forcing full image rebuild"
      shift 1
      ;;
    --usage)
      print_usage
      exit 0
      ;;
    *)
      PARAMS="$PARAMS $1"
      shift 1
      ;;
  esac
done
eval set -- "$PARAMS"

COMPUTE_DOCKERFILE="montecarlo/docker-build/Dockerfile-compute"
BASE_DOCKERFILE="Dockerfile-base"
DATA_DOCKERFILE="Dockerfile-data"
CLIENT_DOCKERFILE="Dockerfile-client"
MODEL_DOCKERFILE="Dockerfile-model"

DOCKER_ARGS=""
if [[ "${FORCE:=}" = 1 ]]; then
  DOCKER_ARGS="${DOCKER_ARGS} --no-cache"
fi

TAG_COMPUTE_LOCAL_NOVERS="$IMAGE-compute"
TAG_COMPUTE_LOCAL="$TAG_COMPUTE_LOCAL_NOVERS:$IMAGE_VERS"
TAG_COMPUTE_REMOTE_NOVERS="$REGISTRY/$TAG_COMPUTE_LOCAL_NOVERS"
TAG_COMPUTE_REMOTE="$REGISTRY/$TAG_COMPUTE_LOCAL"

TAG_DATA_LOCAL_NOVERS="$IMAGE-data"
TAG_DATA_LOCAL="$TAG_DATA_LOCAL_NOVERS:$IMAGE_VERS"
TAG_DATA_REMOTE_NOVERS="$REGISTRY/$TAG_DATA_LOCAL_NOVERS"
TAG_DATA_REMOTE="$REGISTRY/$TAG_DATA_LOCAL"

TAG_CLIENT_LOCAL_NOVERS="$IMAGE-client"
TAG_CLIENT_LOCAL="$TAG_CLIENT_LOCAL_NOVERS:$IMAGE_VERS"

function update_image_tags() {
  # set the image tags in docker-compose.yml to the correct version
  sed -re 's,\$TAG_DATA_REMOTE,'"${TAG_DATA_REMOTE}"',' | \
  sed -re 's,\$TAG_COMPUTE_REMOTE,'"${TAG_COMPUTE_REMOTE}"','
}
function set_timezone() {
  # set the docker-compose.yml timezone
  local TZ="$(timedatectl status | grep 'Time zone' | awk '{print $3}')"
  sed -re 's,\$TZ,'"${TZ}"','
}
generate_compose_file() {
  cat dosecalc/docker-compose.in | \
    update_image_tags   | \
    set_timezone          \
    > dosecalc/docker-compose.yml
}

sudo echo "obtaining sudo rights..."

# Run an automated build of the production docker image used for MC simulation
pushd dosecalc/
  if [[ ${REGEN:=} = 1 ]] || [ ! -f "montecarlo/docker-build/binaries/Application.tgz" ]; then
      mkdir -p montecarlo/docker-build/binaries
      pushd montecarlo/docker-build
        ./build-binaries.sh "$(realpath .)" "$(realpath ..)" "${G4IMAGE}-dev:${G4VERS}"
      popd
  fi
  if [[ ${REGEN:=} = 1 ]] || [ ! -f "${COMPUTE_DOCKERFILE}" ]; then
    montecarlo/docker-build/build-image.sh "${G4IMAGE}:${G4VERS}-data" "${COMPUTE_DOCKERFILE}"
  fi
  sudo docker build --rm --force-rm -t "${TAG_COMPUTE_REMOTE}" -t "${TAG_COMPUTE_LOCAL_NOVERS}" -t "${TAG_COMPUTE_LOCAL}" -f ${COMPUTE_DOCKERFILE} .
  # sudo docker push "${TAG_COMPUTE_REMOTE}" || true
popd

# build containers for pipeline components separate from MC computation
sudo docker build -t "mcdose-base" -f ${BASE_DOCKERFILE} ${DOCKER_ARGS} .
sudo docker build --rm --force-rm -t "${TAG_DATA_REMOTE}" -t "${TAG_DATA_LOCAL_NOVERS}" -t "${TAG_DATA_LOCAL}" -f ${DATA_DOCKERFILE} .
# sudo docker push "${TAG_DATA_REMOTE}" || true

sudo docker build --rm --force-rm -t "${TAG_CLIENT_LOCAL_NOVERS}" -t "${TAG_CLIENT_LOCAL}" -f ${CLIENT_DOCKERFILE} .
sudo docker build --rm --force-rm -t "mcdose" -f ${MODEL_DOCKERFILE} mcdose

echo "Generating docker-compose.yml"
generate_compose_file
echo "If you plan to run in docker swarm, please first push your images to a registry with 'docker-compose push'" \
     "and then deploy the stack with 'docker stack deploy -c docker-compose.yml <stack-name>'"

# cleanup
# rm -rf docker-build/binaries
