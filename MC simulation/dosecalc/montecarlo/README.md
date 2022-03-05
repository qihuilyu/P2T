# Geant4-based Monte Carlo Simulation Module
General monte carlo (MC) dose calculation is provided by the Geant4 particle simulation engine. This module provides a framework specific to the task of voxel-wise dose calculation.

There are two ways to use this module: native or docker-based execution. For most purposes, the docker-based execution should work fine, and is much simpler to install. For more advanced usage, the native installation is preferred but relies on the user's comfortability with manually compling Geant4 first. Instructions for both use cases are provided.

## Usage
### Docker-based Usage
#### Installation
To build all images, simply navigate to `<root>/` and run `./docker-build.sh`  
Afterward, the docker image named "clouddose-compute" can be used with custom `docker run ...` commands, or the convenience script `./docker-exec.sh` can be used instead

#### Execution
To run simulation for one beamlet (manually), browse to `<root>/dosecalc/montecarlo/` and run  
  `./docker-exec.sh --data <input-folder> [args ...]`

`[args ...]` should be replaced by the arguments of the geant4 dose calculation engine, assuming the current working directory (from the docker container's point-of-view) is set to the folder specified using the `--data <input-folder>` parameter. For example, if you have a directory with the following structure:
```bash
mc_data/
├── beamon.in
├── gps.mac
├── init.in
└── mcgeo.txt
```
then the command to run the simulation would be:  
```
./docker-exec.sh --data "./mc_data/" mcgeo.txt init.in gps.mac beamon.in
```
and the outputs of the simulation will be placed inside of `./mc_data/results/`

### Native Usage
COMING SOON

---

## Options & Features
COMING SOON
