#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <mpi.h>
#include "primme.h"   /* header file is required to run primme */

#ifndef min
#  define min(a, b) ((a) < (b) ? (a) : (b))
#endif
/* Buffers...*/
#define SOUTH 0
#define NORTH 1
#define WEST  2
#define EAST  3
#define DOWN  4
#define UP    5
/* Struct to hold Field and MPI buffers*/
typedef struct
{
    complex double*                data;           /* bulk data*/
    complex double*                buffer[6];      /* MPI buffers*/
}
Field;


/* Make local array sizes visible */
int n3l;
int n2l;
int n1l;
/* Make grid sizes visible */
int np3,np2,np1;
/* Timeslice selection */
int x0;
/* Configurations */
complex double *U;



/* MPI Stuff*/
int mpi_dest[6];
int mpi_src [6];

/* Initialise lattice buffers*/
Field alloc_field(void)
{
    int i;
    Field phi;
    
    phi.data = NULL;
    
    /* North and South buffers: length = 3*n3l*n1l */
    phi.buffer[0] = malloc(sizeof(complex double)*3*n3l*n1l);
    for (i=0;i<3*n3l*n1l;i++) phi.buffer[0][i] = 0;
    phi.buffer[1] = malloc(sizeof(complex double)*3*n3l*n1l);
    for (i=0;i<3*n3l*n1l;i++) phi.buffer[1][i] = 0;
    /* East  and West  buffers: length = 3*n2l*n1l */
    phi.buffer[2] = malloc(sizeof(complex double)*3*n2l*n1l);
    for (i=0;i<3*n2l*n1l;i++) phi.buffer[2][i] = 0;
    phi.buffer[3] = malloc(sizeof(complex double)*3*n2l*n1l);
    for (i=0;i<3*n2l*n1l;i++) phi.buffer[3][i] = 0;
    /* Up  and   Down  buffers: length = 3*n3l*n2l */
    phi.buffer[4] = malloc(sizeof(complex double)*3*n3l*n2l);
    for (i=0;i<3*n3l*n2l;i++) phi.buffer[4][i] = 0;
    phi.buffer[5] = malloc(sizeof(complex double)*3*n3l*n2l);
    for (i=0;i<3*n3l*n2l;i++) phi.buffer[5][i] = 0;
    
    return phi;
}

void free_field(Field phi)
{
    int i;
    for (i=0;i<6;i++) free(phi.buffer[i]);
}
/* Function for getting the spinor at neighbouring lattice points. hop 0 for -1 x1 vector, hop 1 for +1 x1 vector, hop 2 for -1 x2 vector...  */
complex double neigh(Field* phi, int site, int hop, int comp)
{
    /* Get (x3,x2,x1)*/
    static const int hop_x3[6] = {0, 0, 0, 0,-1,+1};
    static const int hop_x2[6] = {0, 0,-1,+1, 0, 0};
    static const int hop_x1[6] = {-1,+1, 0, 0, 0,0};
    
    int x1 = site / (3*n3l*n2l);
    int x2 = (site-3*n3l*n2l*x1)/(3*n3l);
    int x3 = (site-3*n3l*n2l*x1-3*n3l*x2) / 3;

    x1+=hop_x1[hop];
    x2+=hop_x2[hop];
    x3+=hop_x3[hop];
    
    
    if (x2==-1 ) return phi->buffer[0][comp+3*x3+3*n3l*x1];
    if (x2==n2l) return phi->buffer[1][comp+3*x3+3*n3l*x1];
    if (x3==-1 ) return phi->buffer[2][comp+3*x2+3*n2l*x1];
    if (x3==n3l) return phi->buffer[3][comp+3*x2+3*n2l*x1];
    if (x1==-1 ) return phi->buffer[4][comp+3*x3+3*n3l*x2];
    if (x1==n1l) return phi->buffer[5][comp+3*x3+3*n3l*x2];
    
    return phi->data[comp+3*x3+3*n3l*x2+3*n3l*n2l*x1];

}
/* Function for getting the Su3 configuration at neighbouring lattice points. hop 0 for -1 x1 vector, hop 1 for +1 x1 vector, hop 2 for -1 x2 vector...  */
complex double config(complex double* U, int site, int hop, int comp, MPI_Comm* comm)
{
    /* Get global (x3,x2,x1)*/
    int rank,p3,p2,p1;
    MPI_Comm_rank(*comm,&rank);
    
    p1=rank/(np3*np2);
    p2 = (rank-np3*np2*p1)/np3;
    p3 = rank % np3;
    
    int ngx3=np3*n3l;
    int ngx2=np2*n2l;
    int ngx1=np1*n1l;
    
    int x1l = site / (3*n3l*n2l);
    int x2l = (site-3*n3l*n2l*x1l)/(3*n3l);
    int x3l = (site-3*n3l*n2l*x1l-3*n3l*x2l) / 3;
    int  c = site%3;
    
    int x3=p3*n3l+x3l;
    int x2=p2*n2l+x2l;
    int x1=p1*n1l+x1l;

    if((x3+x2+x1+x0)%2==0){
        if(hop==0){
            if(x1>0) return conj(U[((x1-1)*ngx2*ngx3/2+x2*ngx3/2+x3/2)*8*9+2*9+c+3*comp]);
            else return conj(U[((ngx1-1)*ngx2*ngx3/2+x2*ngx3/2+x3/2)*8*9+2*9+c+3*comp]);
        }
        if(hop==1){
            if(x1<ngx1-1) return U[((x1+1)*ngx2*ngx3/2+x2*ngx3/2+x3/2)*8*9+3*9+3*c+comp];
            else return U[(x2*ngx3/2+x3/2)*8*9+3*9+3*c+comp];
        }
        if(hop==2){
            if(x2>0) return conj(U[(x1*ngx2*ngx3/2+(x2-1)*ngx3/2+x3/2)*8*9+4*9+c+3*comp]);
            else return conj(U[(x1*ngx2*ngx3/2+(ngx2-1)*ngx3/2+x3/2)*8*9+4*9+c+3*comp]);
        }
        if(hop==3){
            if(x2<ngx2-1) return U[(x1*ngx2*ngx3/2+(x2+1)*ngx3/2+x3/2)*8*9+5*9+3*c+comp];
            else return U[(x1*ngx2*ngx3/2+x3/2)*8*9+5*9+3*c+comp];
        }
        if(hop==4){
            if(x3>0) return conj(U[(x1*ngx2*ngx3/2+x2*ngx3/2+(x3-1)/2)*8*9+6*9+c+3*comp]);
            else return conj(U[(x1*ngx2*ngx3/2+x2*ngx3/2+(ngx3-1)/2)*8*9+6*9+c+3*comp]);
        }
        else{
            if(x3<ngx3-1) return U[(x1*ngx2*ngx3/2+x2*ngx3/2+(x3+1)/2)*8*9+7*9+3*c+comp];
            else return U[(x1*ngx2*ngx3/2+x2*ngx3/2)*8*9+7*9+3*c+comp];
            
        }
    }
    else{
        if(hop==0) return conj(U[(x1*ngx2*ngx3/2+x2*ngx3/2+x3/2)*8*9+3*9+c+3*comp]);
        
        if(hop==1) return U[(x1*ngx2*ngx3/2+x2*ngx3/2+x3/2)*8*9+2*9+3*c+comp];
        
        if(hop==2) return conj(U[(x1*ngx2*ngx3/2+x2*ngx3/2+x3/2)*8*9+5*9+c+3*comp]);
        
        if(hop==3) return U[(x1*ngx2*ngx3/2+x2*ngx3/2+x3/2)*8*9+4*9+3*c+comp];
       
        if(hop==4) return conj(U[(x1*ngx2*ngx3/2+x2*ngx3/2+x3/2)*8*9+7*9+c+3*comp]);
      
        else return U[(x1*ngx2*ngx3/2+x2*ngx3/2+x3/2)*8*9+6*9+3*c+comp];
        
    }
    
}
/* function for importing a configuration into the solver. Might eventually be replaced once I figure out how to access the configs already loaded into ms6*/
void importconfig(char *configname){
    int x3;
    int x2;
    int x1;
    int mu;
    int cc;
    /* get global lattice sizes */
    int nx1g=np1*n1l;
    int nx2g=np2*n2l;
    int nx3g=np3*n3l;
    
    
    FILE *infile;
    char charblock[16];
    infile = fopen(configname,"r");
    for (x1=0; x1<nx1g; x1++) {
        for (x2=0; x2<nx2g; x2++) {
            for (x3=0; x3<nx3g/2; x3++) {
                for (mu=0; mu<8; mu++) {
                    for (cc=0; cc<9; cc++) {
                        fseek(infile, 24+(x0*nx1g*nx2g*nx3g/2+x1*nx2g*nx3g/2+x2*nx3g/2+x3)*8*9*16+mu*9*16+cc*16, SEEK_SET);
                        fread(charblock, 16, 1, infile);
                        U[(x1*nx2g*nx3g/2+x2*nx3g/2+x3)*8*9+mu*9+cc]=*(complex double*)charblock;
                    }
                }
            }
        }
    }
    fclose(infile);
    
}
/* Initialise communication between the processors*/
void comms_init(MPI_Comm* comm)
{
    int rank,p3,p2,p1;
    MPI_Comm_rank(*comm,&rank);
    
    
    p1=rank/(np3*np2);
    p2 = (rank-np3*np2*p1)/np3;
    p3 = rank % np3;
    
    
    mpi_dest[0]=(p3      )    +np3*((p2+1    )%np2)+np3*np2*p1;
    mpi_src [0]=(p3      )    +np3*((p2-1+np2)%np2)+np3*np2*p1;
    
    mpi_dest[1]=mpi_src[0];
    mpi_src [1]=mpi_dest[0];
    
    mpi_dest[2]=(p3+1    )%np3+np3*((p2      )    )+np3*np2*p1;
    mpi_src [2]=(p3-1+np3)%np3+np3*((p2      )    )+np3*np2*p1;
    
    mpi_dest[3]=mpi_src [2];
    mpi_src [3]=mpi_dest[2];
    
    mpi_dest[4]=(p3      )    +np3*((p2    )    )+np3*np2*((p1+1    )%np1);
    mpi_src [4]=(p3      )    +np3*((p2    )    )+np3*np2*((p1-1+np1)%np1);
    
    mpi_dest[5]=mpi_src [4];
    mpi_src [5]=mpi_dest[4];
}

/* Covariant spatial laplacian operator*/
void MatrixMatvec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *err) {
    
    int v;            /* vector index, from 0 to *blockSize-1*/
    int j;          /* local matrix row index, from 0 to nLocal */
    int i;          /* index for the buffers*/
    int c;          /* Color index for the buffers*/
    MPI_Comm comm = *(MPI_Comm *) primme->commInfo;
    
    Field  xvec = alloc_field();
    complex double *yvec;
    complex double *northsouth;
    complex double *eastwest;
    complex double *updown;
    
    /* Buffer for receives*/
    northsouth=malloc(sizeof(complex double)*3*n3l*n1l);
    eastwest=malloc(sizeof(complex double)*3*n2l*n1l);
    updown=malloc(sizeof(complex double)*3*n3l*n2l);
    
    
    MPI_Status status[6];
    MPI_Request req[6];
    for (v=0; v<*blockSize; v++)
    {
        xvec.data = (complex double *)x + *ldx*v;
        yvec      = (complex double *)y + *ldy*v;
        
        /* Receives*/
        MPI_Irecv(xvec.buffer[0], 3*n3l*n1l, MPI_C_DOUBLE_COMPLEX, mpi_src [0], 101, comm,
                  &req[0]);
        MPI_Irecv(xvec.buffer[1], 3*n3l*n1l, MPI_C_DOUBLE_COMPLEX, mpi_src [1], 102, comm,
                  &req[1]);
        MPI_Irecv(xvec.buffer[2], 3*n2l*n1l, MPI_C_DOUBLE_COMPLEX, mpi_src [2], 103, comm,
                  &req[2]);
        MPI_Irecv(xvec.buffer[3], 3*n2l*n1l, MPI_C_DOUBLE_COMPLEX, mpi_src [3], 104, comm,
                  &req[3]);
        MPI_Irecv(xvec.buffer[4], 3*n3l*n2l, MPI_C_DOUBLE_COMPLEX, mpi_src [4], 105, comm,
                  &req[4]);
        MPI_Irecv(xvec.buffer[5], 3*n3l*n2l, MPI_C_DOUBLE_COMPLEX, mpi_src [5], 106, comm,
                  &req[5]);
        
        for (i=0;i<n1l;i++) for (j=0;j<n3l;j++) for (c=0;c<3;c++) northsouth[c+3*j+3*n3l*i] = xvec.data[c+3*j+3*n3l*(n2l-1)+3*n3l*n2l*i];
        MPI_Send(northsouth,3*n3l*n1l, MPI_C_DOUBLE_COMPLEX, mpi_dest[0], 101, comm);
        for (i=0;i<n1l;i++) for (j=0;j<n3l;j++) for (c=0;c<3;c++) northsouth[c+3*j+3*n3l*i] = xvec.data[c+3*j+3*n3l*n2l*i];
        MPI_Send(northsouth,3*n3l*n1l, MPI_C_DOUBLE_COMPLEX, mpi_dest[1], 102, comm);
        for (i=0;i<n1l;i++) for (j=0;j<n2l;j++) for (c=0;c<3;c++) eastwest[c+3*j+3*n2l*i] = xvec.data[c+3*(n3l-1)+3*n3l*j+3*n3l*n2l*i];
        MPI_Send (eastwest, 3*n2l*n1l, MPI_C_DOUBLE_COMPLEX, mpi_dest[2], 103, comm);
        for (i=0;i<n1l;i++) for (j=0;j<n2l;j++) for (c=0;c<3;c++) eastwest[c+3*j+3*n2l*i] = xvec.data[c+3*n3l*j+3*n3l*n2l*i];
        MPI_Send (eastwest, 3*n2l*n1l, MPI_C_DOUBLE_COMPLEX, mpi_dest[3], 104, comm);
        for (i=0;i<n2l;i++) for (j=0;j<n3l;j++) for (c=0;c<3;c++) updown[c+3*j+3*n3l*i] = xvec.data[c+3*j+3*n3l*i+3*n3l*n2l*(n1l-1)];
        MPI_Send (updown,   3*n3l*n2l, MPI_C_DOUBLE_COMPLEX, mpi_dest[4], 105, comm);
        for (i=0;i<n2l;i++) for (j=0;j<n3l;j++) for (c=0;c<3;c++) updown[c+3*j+3*n3l*i] = xvec.data[c+3*j+3*n3l*i];
        MPI_Send (updown,   3*n3l*n2l, MPI_C_DOUBLE_COMPLEX, mpi_dest[5], 106, comm);
        
        MPI_Waitall(6,req,status);
        for (j=0;j<3*n3l*n2l*n1l;j++){
            /* diagonal component*/
            
            yvec[j] = 6.0*xvec.data[j];
            /* off-diagonal components*/
            for (i=0;i<6;i++) for (c=0;c<3;c++) yvec[j]-=neigh(&xvec,j,i,c)*config(U,j,i,c,&comm);
            
        }
        
    }
    free_field(xvec);
    free(northsouth);
    free(eastwest);
    free(updown);
    *err = 0;
}
/* GlobalSum function required by primme*/
static void par_GlobalSum(void *sendBuf, void *recvBuf, int *count,primme_params *primme, int *ierr) {
    MPI_Comm communicator = *(MPI_Comm *)primme->commInfo;
    if (sendBuf == recvBuf) {
        *ierr = MPI_Allreduce(MPI_IN_PLACE, recvBuf, *count, MPI_DOUBLE, MPI_SUM, communicator) != MPI_SUCCESS;
    } else {
        *ierr = MPI_Allreduce(sendBuf, recvBuf, *count, MPI_DOUBLE, MPI_SUM, communicator) != MPI_SUCCESS;
    }
}
/* trace of laplacian from computed eigenvalues, as test. Compare to theory prediction*/
double tracetest(double *evals, int numEvals) {
    double trace;
    int i;
    trace=0;
    for (i=0; i<numEvals; i++) {
        trace+=evals[i];
    }
    return(trace);
}
/* trace of laplacian^2 from computed eigenvalues, as test. Compare to theory prediction*/
double trace2test(double *evals, int numEvals) {
    double trace2;
    int i;
    trace2=0;
    for (i=0; i<numEvals; i++) {
        trace2+=pow(evals[i],2);
    }
    return(trace2);
}
/* trace of laplacian^4 from computed eigenvalues, as test. Compare to theory prediction*/
double trace4test(double *evals, int numEvals) {
    double trace4;
    int i;
    trace4=0;
    for (i=0; i<numEvals; i++) {
        trace4+=pow(evals[i],4);
    }
    return(trace4);
}
/* main function for initialising primme and computing the eigenvalues */
int laplaceeigs (const int dummy[8], char *configname, MPI_Comm *comm, complex double* evecs) {
    
    /* Solver arrays and parameters */
    double *evals;    /* Array with the computed eigenvalues */
    double *rnorms;   /* Array with the computed eigenpairs residual norms */
    primme_params primme;
    /* PRIMME configuration struct */
    
    /* Other miscellaneous items */
    int ret;
    int i;
    double trace;
    double trace4;
    double wt1,wt2;
    
    /* Initialize the infrastructure necessary for communication */

    /* Set default values in PRIMME configuration struct */
    primme_initialize(&primme);
    /* Set problem matrix */
    primme.matrixMatvec = MatrixMatvec;
    /* Function that implements the matrix-vector product
     A*x for solving the problem A*x = l*x */
    
    
    /* Set problem parameters */
    /* Processor grid*/
    np1 = dummy[0]; np2 = dummy[1]; np3 = dummy[2];
    
    /* Local grid size - hard-wired for testing*/
    n1l = dummy[3]; n2l = dummy[4]; n3l = dummy[5];
    x0=dummy[6];
    
    /* My grid communications*/
    comms_init(comm);
    primme.n = 3*(n3l*np3)*(n2l*np2)*(n1l*np1); /* set problem dimension */
    /* Test here - find all*/
    primme.numEvals = dummy[7]; /* Number of wanted eigenpairs */
    primme.eps = 1e-9;     /* ||r|| <= eps * ||matrix|| */
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
    MPI_Comm_size(*comm, &primme.numProcs);
    MPI_Comm_rank(*comm, &primme.procID);
    if (np3*np2*np1 != primme.numProcs)
    {
        if (primme.procID==0) printf(
                                     "Error - number of processors (%d) does not match grid (%d x %d)\n",
                                     primme.numProcs, np3,np2);
        MPI_Finalize();
        return 1;
    }
    
    primme.commInfo = comm; /* User-defined member to pass the communicator to
                              globalSumReal and broadcastReal */
    /* In this example, the matrix is distributed by rows, and the first
     * processes may have an extra row in order to distribute the remaining rows
     * n % numProcs */
    PRIMME_INT nLocal = primme.n / primme.numProcs +
    (primme.n % primme.numProcs > primme.procID ? 1 : 0);
    primme.nLocal = nLocal; /* Number of local rows */
    primme.globalSumReal = par_GlobalSum;
    
    /* Display PRIMME configuration struct (optional) */
    /*if (primme.procID == 0) primme_display_params(primme);*/
    
    /* Allocate space for converged Ritz values and residual norms */
    evals = (double*)malloc(primme.numEvals*sizeof(double));
    rnorms = (double*)malloc(primme.numEvals*sizeof(double));
    U = (complex double*)malloc(12*primme.n*sizeof(complex double));
    
    fprintf(primme.outputFile,"%s",configname);
    MPI_Barrier(MPI_COMM_WORLD);
    wt1=MPI_Wtime();
    importconfig(configname);
    wt2=MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    fprintf(primme.outputFile, "primme importconfig time: %.2e sec\n",wt2-wt1);
    
    
    /* Call primme  */
    ret = zprimme(evals, evecs, rnorms, &primme);
    trace=tracetest(evals, primme.numEvals);
    trace4=trace4test(evals, primme.numEvals);
    
    if (primme.procID == 0) {
        if (ret != 0) {
            fprintf(primme.outputFile,
                    "Error: primme returned with nonzero exit status: %d \n",ret);
            return -1;
        }

         
        /* Reporting (optional) */
        
        /*fprintf(primme.outputFile,"evecs eig: %14.6e \n",creal(evecs[(0*n1l*n2l*n3l+3+2*n3l+1*n2l*n3l)*3+1]));*/
        /*
         for (i=0; i < primme.initSize; i++) {
            fprintf(primme.outputFile, "Eval[%d]: %-22.15E rnorm: %-22.15E\n", i+1,
                    evals[i], rnorms[i]);
        }
         */
        
        fprintf(primme.outputFile, "Trace: %f\n", trace);
        fprintf(primme.outputFile, "Trace4: %f\n", trace4);
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
         
         
    }
    
    primme_free(&primme);
    free(evals);
    free(rnorms);
    free(U);
    
    /* Tear down the communication infrastructure */
    return 0;
}
