PRIMME_DIR=/work/mjp/lucius_test/primme-master
LDFLAGS = -llapack -lblas -lm -O3
INCLUDE := -I${PRIMME_DIR}/include
LIBS := ${PRIMME_DIR}/lib/libprimme.a

CFLAGS := -O3

EXEC = laplacecov

CC = mpicc
CLDR = mpicc
.PHONY: clean

$(EXEC): %: %.o
	$(CLDR) -o $@ $@.o $(LIBDIRS) $(INCLUDE) $(LIBS) $(LDFLAGS) 

%.o : %.c
	$(CC) $(CFLAGS) $(DEFINES) $(INCLUDE) -c $< -o $@

clean:
	@ -rm -f *.o  $(EXEC)