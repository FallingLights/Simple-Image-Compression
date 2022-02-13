#!/bin/bash

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input)
      INPUT="$2"
      shift # past argument
      shift # past value
      ;;
    -k|--kluster)
      KLUSTERS="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--iterations)
      ITERATIONS="$2"
      shift # past argument
      shift # past value
      ;;
    -r|--repeat)
      REPEAT="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters


for ((i = 1 ; i <= $REPEAT ; i++)); do
  srun --reservation=fri ./serial.out ~/Stiskanje-Slike-PS/images/png/bird.png -k $KLUSTERS -i $ITERATIONS  >> serial_out${i}.txt
done

if [[ -n $1 ]]; then
    echo "Last line of file specified as non-opt/last argument:"
    tail -1 "$1"
fi