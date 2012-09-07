include ../common/Makefile.common

MY_SRC_STUB=D2Q9_RegVs_SNL_MB_LBGK_ts

$(MY_SRC_STUB).$(MEXT): $(MY_SRC_STUB).o
	env MATLAB=$(MATLAB) $(NVMEX) $(MY_SRC_STUB).o $(JKTLIB) $(CUDALIB) -cxx -largeArrayDims

$(MY_SRC_STUB).o:  $(MY_SRC_STUB).cu
	$(NVCC) -o $(MY_SRC_STUB).o -c $(MY_SRC_STUB).cu $(CFLAGS) $(PTXFLAGS)

clean:
	rm -f *.$(MEXT) *.o