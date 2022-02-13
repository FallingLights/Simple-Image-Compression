#!/bin/bash

#SBATCH --job-name=serial
#SBATCH --output=results-serial.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --reservation=fri

for CLUSTERS in 4
do
        for ITERS in 10
        do
                for ASD in {1..5}
                do
                        ./serial.out -k $CLUSTERS -i $ITERS -s 12345 ~/Stiskanje-Slike-PS/images/png/bird.png
                done
        done
done