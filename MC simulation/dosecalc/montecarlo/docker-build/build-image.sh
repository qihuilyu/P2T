#!/bin/bash
IMG="andreadotti/geant4"
DOCKERFILE="Dockerfile-compute"
BINLOC="/usr/local/geant4/applications/bin"
LIBLOC="/usr/local/geant4/applications/lib"
if [ "$1" == "-h" ];then
   echo "Usage: $0 [fromimage] [dockerfile] [binloc] [libloc]"
   echo "     fromimage: Docker image to start from (def: $IMG)"
   echo "     dockerfile: Docker filename to generate (def: $DOCKERFILE)"
   echo "     binloc: Location of binaries (def: $BINLOC)"
   echo "     libloc: Location of libraries (def: $LIBLOC)"
   exit 1
fi
[ $# -ge 1 ] && IMG=$1
[ $# -ge 2 ] && DOCKERFILE=$2
[ $# -ge 3 ] && BINLOC=$3
[ $# -ge 4 ] && LIBLOC=$4
cat << EOF > ${DOCKERFILE}
#Auto-generated, changes will be lost if $0 is re-run
FROM $IMG
MAINTAINER Ryan Neph (ryanneph@ucla.edu)
RUN echo 'deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main\ndeb-src http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main' >> /etc/apt/sources.list.d/python3.6.list; \\
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776; \\
    apt update; \\
    apt-get -y install curl python3.6 libprotobuf-dev; \\
    curl https://bootstrap.pypa.io/get-pip.py > /get-pip.py; \\
    apt-get purge -y curl; apt-get autoremove -y; \\
    python3.6 /get-pip.py; \\
    rm -rf /get-pip.py; \\
    python3.6 -m pip install pathvalidate numpy bson scipy; \\
    rm -rf /var/lib/apt/lists/*;
ADD montecarlo/docker-build/binaries/*.tgz /
ENV PATH="$BINLOC:\$PATH"
ENV LD_LIBRARY_PATH="$LIBLOC:\$LD_LIBRARY_PATH"
COPY ./webapi /webapi 
WORKDIR /webapi
ENTRYPOINT [ "/entry-point.sh", "python3.6", "computeserver.py" ]
CMD [ ]
EOF

echo "Generation of Dockerfile, manually adjust it if needed, and then run:"
echo "Manually modify run.sh script if needed"
echo "docker build -f \"${DOCKERFILE}\" -t <image-tag>."
