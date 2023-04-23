PRIMME_DIR=/work/mjp/lucius_test/primme-master
LDFLAGS = -llapack -lblas -lm -O3
INCLUDE := -I${PRIMME_DIR}/include
LIBS := ${PRIMME_DIR}/lib/libprimme.a $(LIBS)

CFLAGS := -O3

EXEC = laplacecov

CC = mpicc
CLDR = mpicc

laplacecov: navigatecov.h

$(EXEC): %: %.o
	$(CLDR) -o $@ $@.o $(LIBDIRS) $(INCLUDE) $(LIBS) $(LDFLAGS) 

.c.o:
	$(CC) $(CFLAGS) $(DEFINES) $(INCLUDE) -c $< -o $@