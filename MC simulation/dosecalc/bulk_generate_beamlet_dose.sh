#!/bin/bash
set -euo pipefail

declare -A map_nparticles
# map_nparticles=( ["2000"]="0075" ["5000"]="0076" ["10000"]="0077" ["20000"]="0078" )
map_nparticles=( ["2000"]="0109" ["5000"]="0110" ["10000"]="0077" ["20000"]="0078" )
DBDATA="/media/hdd2/g4sim/data"
RESULTS="/media/hdd1/g4sim"
LOGFILE='./preddose_summary.log'

while (( $# )); do
  case "$1" in 
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --cpu)
      CPU=1
      shift
      ;;
    --nomount)
      NOMOUNT=1
      shift
      ;;
  esac
done

: ${OVERWRITE:=0} >/dev/null
: ${NOMOUNT:=0} >/dev/null
: ${CPU:=0} >/dev/null
if [[ "${NOMOUNT}" = 1 ]]; then
  nomount_arg="--nomount"
else
  nomount_arg=""
fi
: ${nomount_arg:=${NOMOUNT}} >/dev/null

function gen_beamlet_dose() {
  # PARSE ARGS
  while (( "$#" )); do
    case "$1" in 
      --doi)
        local doi="$2"
        shift 2
        ;;
      --ptv)
        local ptv="$2"
        shift 2
        ;;
      --mag)
        local magfield="$2"
        shift 2
        ;;
      --nparticles)
        local nparticles="$2"
        shift 2
        ;;
      --thresh)
        local drop_thresh="$2"
        shift 2
        ;;
      --ctx|--context)
        local ctx="$2"
        shift 2
        ;;
      --predict)
        local predict=1
        shift
        ;;
      --config)
        local config="$2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done

  # ASSIGN DEFAULT VALUES
  if [[ ! -v config ]]; then
    local config="${map_nparticles[${nparticles}]}"
  fi
  : ${ptv:="."} >/dev/null
  : ${drop_thresh:=0} >/dev/null
  : ${ctx:=16} >/dev/null

  # NAME OUTPUT FILE ACCORDING TO NUMBER OF PARTICLES
  if (( $(awk '{ print +$1 }' <<< ${nparticles}) >= 100000 )); then
    local out="beamlet_dose_true_ctx${ctx}_t${drop_thresh}.h5"
  else
    local out="beamlet_dose_noisy_ctx${ctx}_t${drop_thresh}.h5"
  fi

  # PREDICT OR NOT?
  if [[ "${predict:=0}" = 1 ]]; then
    local out="beamlet_dose_predicted_ctx${ctx}_t${drop_thresh}.h5"
    local predict_args="--predict \
      /result/runs_shenggpu2/${config}/config.yml \
      /result/runs_shenggpu2/${config}/checkpoints/weights.hdf5 \
      /result/runs_shenggpu2/${config}/normstats.json"
  else
    local predict_args="--nopredict /result/runs_shenggpu2/${config}/normstats.json"
  fi
  local fullout="/result/beamlet_dose/${doi}/${magfield}T/${ptv}/${nparticles}/${out}"

  # CPU or GPU?
  if [[ "${CPU}" = 1 ]]; then
    device_args="--cpu"
  else
    device_args=""
  fi


  # LOGGING
  printf "RUNNING: $doi $ptv $magfield $nparticles $config $fullout" | tee -a "$LOGFILE"
  if [[ "$OVERWRITE" = 0 && -f "${fullout/\/result/${RESULTS}}" ]]; then
    printf "   ...   SKIPPED\n" | tee -a "$LOGFILE"
    return
  fi

  # EXECUTION
  ./docker-exec.sh --dbdata "${DBDATA}" --results "${RESULTS}" ${nomount_arg} \
    -- \
    client.py -LDEBUG database \
    --dbhost 127.0.0.1 --dbport 27099 \
    --dbauth root rootpass \
    --data "/dbdata" \
    generate_beamletdose \
    --magnetic_field ${magfield} --nparticles "$nparticles" \
    --xcontext ${ctx} --zcontext ${ctx} --drop_thresh ${drop_thresh} \
    ${predict_args} \
    ${device_args} \
    --out "$fullout" \
    "/result/beamlet_dose/${doi}/${magfield}T/${ptv}/beamlist.txt" 2>&1 | tee -a preddose_stream.log

  # LOGGING
  if [[ $? = 0 ]]; then
    printf "   ...   SUCCESS\n" | tee -a "$LOGFILE"
  else
    printf "   ...   FAILURE\n" | tee -a "$LOGFILE"
  fi
}

# # HN011
# for nparticles in 5000; do
#   for ctx in 16; do
#     for thresh in "None"; do
#       gen_beamlet_dose --doi HN011 --ptv '.' --mag 1.5 --nparticles ${nparticles} --ctx ${ctx} --thresh "${thresh}" --predict
#       gen_beamlet_dose --doi HN011 --ptv '.' --mag 1.5 --nparticles 1e5           --ctx ${ctx} --thresh "${thresh}" --config ${map_nparticles[${nparticles}]}
#       gen_beamlet_dose --doi HN011 --ptv '.' --mag 0.0 --nparticles 1e5           --ctx ${ctx} --thresh "${thresh}" --config ${map_nparticles[${nparticles}]}
#       gen_beamlet_dose --doi HN011 --ptv '.' --mag 1.5 --nparticles ${nparticles} --ctx ${ctx} --thresh "${thresh}"
#       echo ''
#     done
#   done
# done

# HN010
for ptv in PTV_5400 PTV_5940; do
  for nparticles in 2000 5000; do
    for thresh in "None"; do
      gen_beamlet_dose --doi HN010 --ptv ${ptv} --mag 1.5 --nparticles ${nparticles} --thresh "${thresh}" --predict
      gen_beamlet_dose --doi HN010 --ptv ${ptv} --mag 1.5 --nparticles 1e5           --thresh "${thresh}" --config ${map_nparticles[${nparticles}]}
      gen_beamlet_dose --doi HN010 --ptv ${ptv} --mag 0.0 --nparticles 1e5           --thresh "${thresh}" --config ${map_nparticles[${nparticles}]}
      gen_beamlet_dose --doi HN010 --ptv ${ptv} --mag 1.5 --nparticles ${nparticles} --thresh "${thresh}"
      echo ''
    done
  done
done
