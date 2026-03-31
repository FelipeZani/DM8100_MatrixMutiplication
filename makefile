serial:
	gcc -fopenmp -o serialMux serialMux.c
omp:
	gcc -fopenmp -o openMPMux openMPMux.c

runall : serial omp
	./serialMux
	./openMPMux
clean:
	rm ${PROG}
