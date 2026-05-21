serial:
	gcc -fopenmp -o serialMux serialMux.c src/matrixTools.c
omp:
	gcc -fopenmp -o openMPMux openMPMux.c src/matrixTools.c
mpi:
	mpicc -o mpiMux mpiMux.c src/matrixTools.c

cuda:
	nvcc -o cudaMux cudaMux.cu src/matrixTools.c

runall : serial omp mpi cuda
	./serialMux
	./openMPMux
	mpirun -n 1 ./mpiMux
	./cudaMux 
clean:
	rm ${PROG}
