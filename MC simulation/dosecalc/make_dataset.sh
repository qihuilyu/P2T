#!/bin/bash
set -euo pipefail

while (( $# )); do
  case "$1" in 
    --overwrite)
      OVERWRITE=1
      shift
      ;;
  esac
done

DBDATA="/media/hdd2/g4sim/data"
RESULTS="/media/hdd1/g4sim"
LOGFILE='./make_dataset_summary.log'
: ${OVERWRITE:=0} >/dev/null

function generate_dataset() {
  # PARSE ARGS
  while (( "$#" )); do
    case "$1" in 
      --context|--ctx)
        local context="$2"
        shift 2
        ;;
      --nparticles)
        local nparticles="$2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done

  date="$(date +%Y%b%d | tr '[:upper:]' '[:lower:]')"
  outfolder="/result/traindata/traindata_${date}_${nparticles%000}k_${context}ctx"

  # LOGGING
  printf "RUNNING: $nparticles $context $outfolder" | tee -a "$LOGFILE"
  if ! [[ "$OVERWRITE" = 0  && -d "${outfolder/\/result/${RESULTS}}" ]]; then
    # create dataset
    sudo ./docker-exec.sh \
      --dbdata "${DBDATA}" \
      --results "${RESULTS}" \
      -- \
      client.py -LDEBUG database \
      --dbhost 127.0.0.1 --dbport 27099 \
      --dbauth root rootpass \
      --data /dbdata \
      generate_dataset \
        --nparticles ${nparticles} \
        --out "${outfolder}" \
        --xcontext ${context} --zcontext ${context} --fsize 2
  else
    printf "   ...   SKIPPED (gen)\n" | tee -a "$LOGFILE"
  fi

  if ! [[ "$OVERWRITE" = 0  && -d "${outfolder/\/result/${RESULTS}}_norm" ]]; then
    # normalize dataset
    sudo ./docker-exec.sh \
      --dbdata "${DBDATA}" \
      --results "${RESULTS}" \
      -- \
      normalize.py \
        "${outfolder}" \
        "${outfolder}_norm"
  else
    printf "   ...   SKIPPED (norm)\n" | tee -a "$LOGFILE"
  fi

  # LOGGING
  if [[ $? = 0 ]]; then
    printf "   ...   SUCCESS\n" | tee -a "$LOGFILE"
  else
    printf "   ...   FAILURE\n" | tee -a "$LOGFILE"
  fi
}

context=14
for nparticles in 2000 5000; do
  generate_dataset --nparticles ${nparticles} --context ${context}
done
