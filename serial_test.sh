#!/bin/bash

#SBATCH --job-name=serial
#SBATCH --output=results-serial.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --reservation=fri

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

rm serial_out*
for ((i = 1 ; i <= $REPEAT ; i++)); do
  ./serial.out ~/Stiskanje-Slike-PS/images/png/bird.png -k $KLUSTERS -i $ITERATIONS -s 12345  > serial_out${i}.txt
done

SUM=0
rm times.txt
for ((i = 1 ; i <= $REPEAT ; i++)); do
    TEMP=$(awk -F':' '{print $2}' serial_out${i}.txt | grep -o '[0-9.]\+' | awk 'NR==9 {print $1}')
    echo $TEMP >> times.txt
    #SUM=$(($SUM + $TEMP))
done

sed -i 's/./,/' times.txt
#OUT=$(($SUM / $REPEAT))
#echo $OUT
