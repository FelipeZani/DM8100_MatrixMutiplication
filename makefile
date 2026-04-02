serial:
	gcc -fopenmp -o serialMux serialMux.c src/matrixTools.c
omp:
	gcc -fopenmp -o openMPMux openMPMux.c src/matrixTools.c
mpi:
	mpicc -o mpiMux mpiMux.c src/matrixTools.c

runall : serial omp mpi
	./serialMux
	./openMPMux
	mpirun -n 1 ./mpiMux
clean:
	rm ${PROG}
