serial:
	gcc -fopenmp -O3 -ffast-math -mavx2 -o serialMux serialMux.c src/matrixTools.c
omp:
	gcc -fopenmp -O3 -ffast-math -mavx2 -o openMPMux openMPMux.c src/matrixTools.c 

mpi:
	mpicc -o mpiMux mpiMux.c src/matrixTools.c

cuda:
	nvcc -o cudaMux cudaMux.cu src/matrixTools.c

runall : serial omp mpi cuda
	./serialMux 1024
	./openMPMux 1024
	mpirun -n 1 ./mpiMux
	./cudaMux 
