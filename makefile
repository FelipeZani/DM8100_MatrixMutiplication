serial:
	gcc -fopenmp -o serialMux serialMux.c src/matrixTools.c
omp:
	gcc -fopenmp -o openMPMux openMPMux.c src/matrixTools.c

runall : serial omp
	./serialMux
	./openMPMux
clean:
	rm ${PROG}
