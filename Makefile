CC = gcc
CC_FLAGS = -Wall
CC_OMP = -fopenmp

all: serial.out omp.out

clean:
	rm serial.out omp.out result.jpg

serial.out: Serial/serial.c
	$(CC) $(CC_FLAGS) -o serial.out Serial/serial.c -lm -fopenmp -O2

omp.out: OpenMP/OpenMP.c
	$(CC) $(CC_FLAGS) $(CC_OMP) -o omp.out OpenMP/OpenMP.c -lm -fopenmp -O2