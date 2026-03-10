serial:
	gcc -o serialMux serialMux.c
	./serialMux
clean:
	rm ${PROG}
