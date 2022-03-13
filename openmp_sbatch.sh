#!/bin/bash

#SBATCH --job-name=openmp3
#SBATCH --output=results-mp3.log
#SBATCH --ntasks=64
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --reservation=fri

for THREADS in 4 8 16
do
        for CLUSTERS in 4 8 16 32 64 
        do
                for ITERS in 20
                do
                        for ASD in {1..20}
                        do
                                ./omp.out -k $CLUSTERS -i $ITERS -t $THREADS -s 12345 -o resoult_omp.png ~/Stiskanje-Slike-PS/images/png/bird.png
                        done
                done
        done
done