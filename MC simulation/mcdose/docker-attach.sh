#!/bin/bash

function get_nvidia_runtime_args() {
  # emit the correct nvidia-docker command line arguments based on the docker version
  if sudo docker run --help | grep -e '--gpus' 2>&1 >/dev/null; then
    echo '--gpus=all'
  else
    echo '--runtime=nvidia'
  fi
}

sudo docker run --rm \
    $(get_nvidia_runtime_args) \
    -u $(id -u) \
    -e TZ="$(timedatectl status | grep 'Time zone' | awk '{print $3}')" \
    -w "/src" \
    -v "$(realpath ./mcdose):/src/mcdose" \
    -it  ${@} --entrypoint /bin/bash \
    mcdose
