#!/bin/bash
set -e
#Name of the output tarball
[ "$APPTARNAME" == "" ] && export APPTARNAME=Application.tgz
#By default install in this location so we stup PATH
#and LD_LIBRARY_PATH automatically
[ "$APPINSTDIR" == "" ] && export APPINSTDIR=/usr/local/geant4/applications
#Pass extra options
xopts=""
[ "$APPINSTDIR" != "" ] && xopts="${xopts} -DCMAKE_INSTALL_PREFIX=${APPINSTDIR}"
for val in "$@";do
  xopts="${xopts} ${val}"
done
echo "Geant4 version: "`geant4-config --version`
echo "Application tarball output name: "${APPTARNAME}
echo "Application image installation area: "${APPINSTDIR}

# make local copy of source based on current copy from host system
src="/App-src-cp"
cp -r /App-src "$src"

# compile protobufs + make available to user app
ls $src/protobufs/*.protobuf && {
  echo "Compiling protobufs..."
  apt update
  apt install -y libprotobuf-dev protobuf-compiler
  pushd "$src" >/dev/null
    pushd protobufs >/dev/null
      pbout='/tmp/protobuf-generated-cpp'
      mkdir -p "$pbout"
      protoc --cpp_out="$pbout" *.protobuf
      cp $pbout/*.protobuf.pb.h  "$src/include"
      cp $pbout/*.protobuf.pb.cc "$src/src"
    popd >/dev/null
  popd >/dev/null
}

cmake -DGeant4_DIR=/usr/local/geant4/lib/Geant4-* \
      ${xopts} "$src"

make -j`nproc` 
make install/fast
tar -czf /build/binaries/${APPTARNAME} ${APPINSTDIR}

