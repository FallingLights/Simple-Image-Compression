#!/bin/bash

#SBATCH --job-name=serial2
#SBATCH --output=results-serial2.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --reservation=fri

for CLUSTERS in 4 8 16 32 64
do
        for ITERS in 20
        do
                for ASD in {1..20}
                do
                        ./serial.out -k $CLUSTERS -i $ITERS -s 12345 ~/Stiskanje-Slike-PS/images/png/bird.png
                done
        done
done