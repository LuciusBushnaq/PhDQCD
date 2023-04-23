#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

#include "primme.h"   /* header file is required to run primme */

#ifndef min
#  define min(a, b) ((a) < (b) ? (a) : (b))
#endif





/* Make local array sizes visible*/
int nxl;
int nyl;
#include "navigate_complex_test.h"


/* MPI Stuff*/
int mpi_dest[4];
int mpi_src [4];


void comms_init(int npx, int npy)
{
    int rank,px,py;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
    
    
    px = rank % npx;
    py = rank / npx;
    

    mpi_dest[0]=(px      )    +npx*((py+1    )%npy);
    mpi_src [0]=(px      )    +npx*((py-1+npy)%npy);

    mpi_dest[1]=(px      )    +npx*((py-1+npy)%npy);
    mpi_src [1]=(px      )    +npx*((py+1    )%npy);

    mpi_dest[2]=(px+1    )%npx+npx*((py      )    );
    mpi_src [2]=(px-1+npx)%npx+npx*((py      )    );

    mpi_dest[3]=(px-1+npx)%npx+npx*((py      )    );
    mpi_src [3]=(px+1    )%npx+npx*((py      )    );
    
    
#if 0
    /* Simple-minded test communication - hard-wired 2x2 grid*/
    switch (rank)
    {
        case 0:
            mpi_dest[0]=2; mpi_dest[1]=2; mpi_dest[2]=1; mpi_dest[3]=1;
            mpi_src [0]=2; mpi_src [1]=2; mpi_src [2]=1; mpi_src [3]=1;
            break;
        case 1:
            mpi_dest[0]=3; mpi_dest[1]=3; mpi_dest[2]=0; mpi_dest[3]=0;
            mpi_src [0]=3; mpi_src [1]=3; mpi_src [2]=0; mpi_src [3]=0;
            break;
        case 2:
            mpi_dest[0]=0; mpi_dest[1]=0; mpi_dest[2]=3; mpi_dest[3]=3;
            mpi_src [0]=0; mpi_src [1]=0; mpi_src [2]=3; mpi_src [3]=3;
            break;
        case 3:
            mpi_dest[0]=1; mpi_dest[1]=1; mpi_dest[2]=2; mpi_dest[3]=2;
            mpi_src [0]=1; mpi_src [1]=1; mpi_src [2]=2; mpi_src [3]=2;
            break;
    }
#endif
}


void MatrixMatvec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *err) {
    
    int i;            /* vector index, from 0 to *blockSize-1*/
    int j;          /* local matrix row index, from 0 to nLocal */
    
    
    
    
    Field  xvec = alloc_field();
    complex double *yvec,*eastwest;
    
    /* Buffer for east-west receives*/
    eastwest=malloc(sizeof(complex double)* nyl);
    
    
    MPI_Status status[4];
    MPI_Request req[4];
    
    
    for (i=0; i<*blockSize; i++)
    {
        xvec.data = (complex double *)x + *ldx*i;
        yvec      = (complex double *)y + *ldy*i;
        
     /* Receives*/
        MPI_Irecv(xvec.buffer[1], nxl, MPI_C_DOUBLE_COMPLEX, mpi_src [0], 101, MPI_COMM_WORLD,
                  &req[0]);
        MPI_Irecv(xvec.buffer[0], nxl, MPI_C_DOUBLE_COMPLEX, mpi_src [1], 102, MPI_COMM_WORLD,
                  &req[1]);
        MPI_Irecv(xvec.buffer[3], nyl, MPI_C_DOUBLE_COMPLEX, mpi_src [2], 103, MPI_COMM_WORLD,
                  &req[2]);
        MPI_Irecv(xvec.buffer[2], nyl, MPI_C_DOUBLE_COMPLEX, mpi_src [3], 104, MPI_COMM_WORLD,
                  &req[3]);
        
        
        MPI_Send(xvec.data,            nxl,
                 MPI_C_DOUBLE_COMPLEX, mpi_dest[0], 101, MPI_COMM_WORLD);
        MPI_Send(xvec.data+nxl*(nyl-1),nxl,
                 MPI_C_DOUBLE_COMPLEX, mpi_dest[1], 102, MPI_COMM_WORLD);
        
        for (j=0;j<nyl;j++) eastwest[j] = xvec.data[nxl*j];
        MPI_Send (eastwest,       nyl, MPI_C_DOUBLE_COMPLEX, mpi_dest[2], 103, MPI_COMM_WORLD);
        for (j=0;j<nyl;j++) eastwest[j] = xvec.data[nxl-1+nxl*j];
        MPI_Send (eastwest,       nyl, MPI_C_DOUBLE_COMPLEX, mpi_dest[3], 104, MPI_COMM_WORLD);
        
        
        MPI_Waitall(4,req,status);
        
        
        
        
        for (j=0;j<nxl*nyl;j++)
        yvec[j] = xvec.data[j] - 0.25 *
        (   neigh(&xvec,j,0) + neigh(&xvec,j,1)
         + neigh(&xvec,j,2) + neigh(&xvec,j,3) );
    }
    
    free_field(xvec);
    free(eastwest);
    *err = 0;
}

static void par_GlobalSum(void *sendBuf, void *recvBuf, int *count,
                          primme_params *primme, int *ierr) {
    MPI_Comm communicator = *(MPI_Comm *) primme->commInfo;
    
    if (sendBuf == recvBuf) {
        *ierr = MPI_Allreduce(MPI_IN_PLACE, recvBuf, *count, MPI_DOUBLE, MPI_SUM, communicator) != MPI_SUCCESS;
    } else {
        *ierr = MPI_Allreduce(sendBuf, recvBuf, *count, MPI_DOUBLE, MPI_SUM, communicator) != MPI_SUCCESS;
    }
}



int main (int argc, char *argv[]) {
    
    /* Solver arrays and parameters */
    double *evals;    /* Array with the computed eigenvalues */
    double *rnorms;   /* Array with the computed eigenpairs residual norms */
    complex double *evecs;    /* Array with the computed eigenvectors;
                      first vector starts in evecs[0],
                      second vector starts in evecs[primme.n],
                      third vector starts in evecs[primme.n*2]...  */
    primme_params primme;
    /* PRIMME configuration struct */
    
    /* Other miscellaneous items */
    int npx,npy;
    int ret;
    int i;
    
    if (argc != 3)
    {
        printf("Usage: %s <npx> <npy> --- specify processor grid dims\n",argv[0]);
        return 1;
    }
    /* Initialize the infrastructure necessary for communication */
    MPI_Init(&argc, &argv);
    
    /* Set default values in PRIMME configuration struct */
    primme_initialize(&primme);
    
    /* Set problem matrix */
    primme.matrixMatvec = MatrixMatvec;
    /* Function that implements the matrix-vector product
     A*x for solving the problem A*x = l*x */
    
    
    /* Set problem parameters */
    
    /* Processor grid*/
    npx = atoi(argv[1]); npy = atoi(argv[2]);
    
    /* My grid communications*/
    comms_init(npx,npy);
    /* Local grid size - hard-wired for testing*/
    nxl = 100 / npx; nyl = 100 / npy;
    
    
    primme.n = (nxl*npx)*(nyl*npy); /* set problem dimension */
    /* Test here - find all*/
    primme.numEvals = 20; /* Number of wanted eigenpairs */
    
    primme.eps = 1e-9;      /* ||r|| <= eps * ||matrix|| */
    primme.target = primme_smallest;
    /* Wanted the smallest eigenvalues */
    
    /* Set advanced parameters if you know what are you doing (optional) */
    /*
     primme.maxBasisSize = 14;
     primme.minRestartSize = 4;
     primme.maxBlockSize = 1;
     primme.maxMatvecs = 1000;
     */
    
    /* Set method to solve the problem */
    primme_set_method(PRIMME_DEFAULT_MIN_MATVECS, &primme);
    /* DYNAMIC uses a runtime heuristic to choose the fastest method between
     PRIMME_DEFAULT_MIN_TIME and PRIMME_DEFAULT_MIN_MATVECS. But you can
     set another method, such as PRIMME_LOBPCG_OrthoBasis_Window, directly */
    
    /* Set parallel parameters */
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &primme.numProcs);
    MPI_Comm_rank(comm, &primme.procID);
    
    if (npx*npy != primme.numProcs)
    {
        if (primme.procID==0) printf(
                                     "Error - number of processors (%d) does not match grid (%d x %d)\n",
                                     primme.numProcs, npx,npy);
        MPI_Finalize();
        return 1;
    }
    
    primme.commInfo = &comm; /* User-defined member to pass the communicator to
                              globalSumReal and broadcastReal */
    /* In this example, the matrix is distributed by rows, and the first
     * processes may have an extra row in order to distribute the remaining rows
     * n % numProcs */
    PRIMME_INT nLocal = primme.n / primme.numProcs +
    (primme.n % primme.numProcs > primme.procID ? 1 : 0);
    primme.nLocal = nLocal; /* Number of local rows */
    primme.globalSumReal = par_GlobalSum;
    
    /* Display PRIMME configuration struct (optional) */
    if (primme.procID == 0) primme_display_params(primme);
    
    /* Allocate space for converged Ritz values and residual norms */
    evals = (double*)malloc(primme.numEvals*sizeof(double));
    evecs = (complex double*)malloc(primme.n*primme.numEvals*sizeof(complex double));
    rnorms = (double*)malloc(primme.numEvals*sizeof(double));
    
    /* Call primme  */
    ret = zprimme(evals, evecs, rnorms, &primme);
    
    double tr=0;
    
    if (primme.procID == 0) {
        if (ret != 0) {
            fprintf(primme.outputFile,
                    "Error: primme returned with nonzero exit status: %d \n",ret);
            return -1;
        }
        
        /* Reporting (optional) */
        for (i=0; i < primme.initSize; i++) {
            tr += evals[i];
            fprintf(primme.outputFile, "Eval[%d]: %-22.15E rnorm: %-22.15E\n", i+1,
                    evals[i], rnorms[i]);
        }
        fprintf(primme.outputFile, " %d eigenpairs converged\n", primme.initSize);
        fprintf(primme.outputFile, "Tolerance : %-22.15E\n",
                primme.aNorm*primme.eps);
        fprintf(primme.outputFile, "Iterations: %-" PRIMME_INT_P "\n",
                primme.stats.numOuterIterations);
        fprintf(primme.outputFile, "Restarts  : %-" PRIMME_INT_P "\n", primme.stats.numRestarts);
        fprintf(primme.outputFile, "Matvecs   : %-" PRIMME_INT_P "\n", primme.stats.numMatvecs);
        fprintf(primme.outputFile, "Preconds  : %-" PRIMME_INT_P "\n", primme.stats.numPreconds);
        fprintf(primme.outputFile, "Orthogonalization Time : %g\n", primme.stats.timeOrtho);
        fprintf(primme.outputFile, "Matvec Time            : %g\n", primme.stats.timeMatvec);
        fprintf(primme.outputFile, "GlobalSum Time         : %g\n", primme.stats.timeGlobalSum);
        fprintf(primme.outputFile, "Broadcast Time         : %g\n", primme.stats.timeBroadcast);
        fprintf(primme.outputFile, "Total Time             : %g\n", primme.stats.elapsedTime);
        if (primme.stats.lockingIssue) {
            fprintf(primme.outputFile, "\nA locking problem has occurred.\n");
            fprintf(primme.outputFile,
                    "Some eigenpairs do not have a residual norm less than the tolerance.\n");
            fprintf(primme.outputFile,
                    "However, the subspace of evecs is accurate to the required tolerance.\n");
        }
        
        switch (primme.dynamicMethodSwitch) {
            case -1: fprintf(primme.outputFile,
                             "Recommended method for next run: DEFAULT_MIN_MATVECS\n"); break;
            case -2: fprintf(primme.outputFile,
                             "Recommended method for next run: DEFAULT_MIN_TIME\n"); break;
            case -3: fprintf(primme.outputFile,
                             "Recommended method for next run: DYNAMIC (close call)\n"); break;
        }
        printf("trace = %lf\n",tr);
    }
    
    primme_free(&primme);
    free(evals);
    free(evecs);
    free(rnorms);
    
    /* Tear down the communication infrastructure */
    MPI_Finalize();
    
    return(0);
}
