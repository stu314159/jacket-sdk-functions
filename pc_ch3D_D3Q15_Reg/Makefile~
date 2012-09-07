include ../common/Makefile.common

MY_SRC_STUB=pc_channel3D_D3Q15_jkt

$(MY_SRC_STUB).$(MEXT): $(MY_SRC_STUB).o
	env MATLAB=$(MATLAB) $(NVMEX) $(MY_SRC_STUB).o $(JKTLIB) $(CUDALIB) -cxx -largeArrayDims

$(MY_SRC_STUB).o:  $(MY_SRC_STUB).cu
	$(NVCC) -o $(MY_SRC_STUB).o -c $(MY_SRC_STUB).cu $(CFLAGS) $(PTXFLAGS)

clean:
	rm -f *.$(MEXT) *.o