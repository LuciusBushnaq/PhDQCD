#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include "primme.h"   /* header file is required to run primme */

void LaplacianMatrixMatvec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *err);
void importconfig();
double tracetest();
double trace4test();
int x0;
int Nx;
complex double *U;
int main (int argc, char *argv[]) {
    /* Solver arrays and parameters */
    double  *evals;    /* Array with the computed eigenvalues */
    double  *rnorms;   /* Array with the computed eigenpairs residual norms */
    complex double  *evecs;    /* Array with the computed eigenvectors;
                                first vector starts in evecs[0],
                                second vector starts in evecs[primme.n],
                                third vector starts in evecs[primme.n*2]...  */
    primme_params primme;
    /* PRIMME configuration struct */
    
    /* Other miscellaneous items */
    int ret;
    int i;
    double trace;
    double trace4;
    x0=0;
    
    /* Set default values in PRIMME configuration struct */
    primme_initialize(&primme);
    /* Set problem matrix */
    primme.matrixMatvec = LaplacianMatrixMatvec;
    /* Function that implements the matrix-vector product
     *                                A*x for solving the problem A*x = l*x */
    
    /* Set problem parameters */
    Nx=4;
    primme.n = Nx*Nx*Nx*3; /* set problem dimension */
    
    
    primme.numEvals = 192;   /* Number of wanted eigenpairs */
    primme.eps = 1e-9;      /* ||r|| <= eps * ||matrix|| */
    primme.target = primme_smallest;
    /* Wanted the smallest eigenvalues */
    
    
    /* Set advanced parameters if you know what are you doing (optional) */
    /*
     *     primme.maxBasisSize = 14;
     *         primme.minRestartSize = 4;
     *             primme.maxBlockSize = 1;
     *                 primme.maxMatvecs = 1000;
     *                     */
    
    /* Set method to solve the problem */
    primme_set_method(PRIMME_DYNAMIC, &primme);
    /* DYNAMIC uses a runtime heuristic to choose the fastest method between
     *         PRIMME_DEFAULT_MIN_TIME and PRIMME_DEFAULT_MIN_MATVECS. But you can
     *                 set another method, such as PRIMME_LOBPCG_OrthoBasis_Window, directly */
    
    
    /* Display PRIMME configuration struct (optional) */
    primme_display_params(primme);
    /* Allocate space for converged Ritz values and residual norms */
    evals = (double*)malloc(primme.numEvals*sizeof(double));
    evecs = (complex double*)malloc(primme.n*primme.numEvals*sizeof(complex double));
    rnorms = (double*)malloc(primme.numEvals*sizeof(double));
    U = (complex double*)malloc(12*primme.n*sizeof(complex double));
    importconfig();
    /* Call primme  */
    ret =  zprimme(evals, evecs, rnorms, &primme);
    
    if (ret != 0) {
        fprintf(primme.outputFile,
                "Error: primme returned with nonzero exit status: %d \n",ret);
        return -1;
    }
    trace=tracetest(evals, primme.numEvals);
    trace4=trace4test(evals, primme.numEvals);
    
    /* Reporting (optional) */
    for (i=0; i < primme.initSize; i++) {
        fprintf(primme.outputFile, "Eval[%d]: %-22.15E rnorm: %-22.15E\n", i+1,
                evals[i], rnorms[i]);
    }
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
    
    
    primme_free(&primme);
    free(evals);
    free(evecs);
    free(rnorms);
    free(U);
    
    
    return(0);
}

void importconfig(){
    int j;
    int k;
    int l;
    int mu;
    int cc;
    FILE *infile;
    char charblock[16];
    infile = fopen("4x4config1n4","r");
    for (l=0; l<Nx; l++) {
        for (k=0; k<Nx; k++) {
            for (j=0; j<Nx/2; j++) {
                for (mu=0; mu<8; mu++) {
                    for (cc=0; cc<9; cc++) {
                        fseek(infile, 24+(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j)*8*9*16+mu*9*16+cc*16, SEEK_SET);
                        fread(charblock, 16, 1, infile);
                        U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j)*8*9+mu*9+cc]=*(complex double*)charblock;
                    }
                }
            }
        }
    }
    fclose(infile);
    
    
}

void LaplacianMatrixMatvec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *err) {
    
    int i;            /* vector index, from 0 to *blockSize-1*/
    int j;
    int k;
    int l;
    int c;
    int row;          /* Laplacian matrix row index, from 0 to matrix dimension */
    complex double *xvec;     /* pointer to i-th input vector x */
    complex double *yvec;     /* pointer to i-th output vector y */
    
    for (i=0; i<*blockSize; i++) {
        xvec = (complex double *)x + *ldx*i;
        yvec = (complex double *)y + *ldy*i;
        for (l=0; l<Nx; l++) {
            for (k=0; k<Nx; k++) {
                for (j=0; j<Nx; j++) {
                    for (c=0; c<3; c++) {
                        row=(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+j)*3+c;
                        yvec[row] = 0.0;
                        yvec[row] += 6.0*xvec[row];
                        if((j+k+l)%2==0){
                            if (j-1 >= 0){
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(j-1))*3]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+(j-1)/2)*8*9+6*9+c])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(j-1))*3+1]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+(j-1)/2)*8*9+6*9+c+3])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(j-1))*3+2]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+(j-1)/2)*8*9+6*9+c+6]);
                            }
                            
                            else {
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(Nx-1))*3]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+(Nx-1)/2)*8*9+6*9+c])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(Nx-1))*3+1]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+(Nx-1)/2)*8*9+6*9+c+3])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(Nx-1))*3+2]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+(Nx-1)/2)*8*9+6*9+c+6]);
                            }
                            
                            if (j+1 < Nx){
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(j+1))*3]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+(j+1)/2)*8*9+7*9+c])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(j+1))*3+1]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+(j+1)/2)*8*9+7*9+c+3])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(j+1))*3+2]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+(j+1)/2)*8*9+7*9+c+6]);
                            }
                            else {
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx)*3]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2)*8*9+7*9+c])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx)*3+1]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2)*8*9+7*9+c+3])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx)*3+2]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2)*8*9+7*9+c+6]);
                            }
                            if (k-1 >= 0){
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(k-1)*Nx+j)*3]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+(k-1)*Nx/2+j/2)*8*9+4*9+c])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(k-1)*Nx+j)*3+1]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+(k-1)*Nx/2+j/2)*8*9+4*9+c+3])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(k-1)*Nx+j)*3+2]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+(k-1)*Nx/2+j/2)*8*9+4*9+c+6]);
                            }
                            
                            else {
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(Nx-1)*Nx+j)*3]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+(Nx-1)*Nx/2+j/2)*8*9+4*9+c])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(Nx-1)*Nx+j)*3+1]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+(Nx-1)*Nx/2+j/2)*8*9+4*9+c+3])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(Nx-1)*Nx+j)*3+2]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+(Nx-1)*Nx/2+j/2)*8*9+4*9+c+6]);
                            }
                            
                            if (k+1 < Nx){
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(k+1)*Nx+j)*3]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+(k+1)*Nx/2+j/2)*8*9+5*9+c])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(k+1)*Nx+j)*3+1]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+(k+1)*Nx/2+j/2)*8*9+5*9+c+3])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(k+1)*Nx+j)*3+2]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+(k+1)*Nx/2+j/2)*8*9+5*9+c+6]);
                            }
                            
                            else {
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+j)*3]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+j/2)*8*9+5*9+c])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+j)*3+1]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+j/2)*8*9+5*9+c+3])-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+j)*3+2]*conj(U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+j/2)*8*9+5*9+c+6]);
                            }
                            
                            if (l-1 >= 0) {
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+(l-1)*Nx*Nx+k*Nx+j)*3]*conj(U[(x0*Nx*Nx*Nx/2+(l-1)*Nx*Nx/2+k*Nx/2+j/2)*8*9+2*9+c])-1.0*xvec[(x0*Nx*Nx*Nx+(l-1)*Nx*Nx+k*Nx+j)*3+1]*conj(U[(x0*Nx*Nx*Nx/2+(l-1)*Nx*Nx/2+k*Nx/2+j/2)*8*9+2*9+c+3])-1.0*xvec[(x0*Nx*Nx*Nx+(l-1)*Nx*Nx+k*Nx+j)*3+2]*conj(U[(x0*Nx*Nx*Nx/2+(l-1)*Nx*Nx/2+k*Nx/2+j/2)*8*9+2*9+c+6]);
                            }
                            
                            else {
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+(Nx-1)*Nx*Nx+k*Nx+j)*3]*conj(U[(x0*Nx*Nx*Nx/2+(Nx-1)*Nx*Nx/2+k*Nx/2+j/2)*8*9+2*9+c])-1.0*xvec[(x0*Nx*Nx*Nx+(Nx-1)*Nx*Nx+k*Nx+j)*3+1]*conj(U[(x0*Nx*Nx*Nx/2+(Nx-1)*Nx*Nx/2+k*Nx/2+j/2)*8*9+2*9+c+3])-1.0*xvec[(x0*Nx*Nx*Nx+(Nx-1)*Nx*Nx+k*Nx+j)*3+2]*conj(U[(x0*Nx*Nx*Nx/2+(Nx-1)*Nx*Nx/2+k*Nx/2+j/2)*8*9+2*9+c+6]);
                            }
                            
                            if (l+1 < Nx){
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+(l+1)*Nx*Nx+k*Nx+j)*3]*conj(U[(x0*Nx*Nx*Nx/2+(l+1)*Nx*Nx/2+k*Nx/2+j/2)*8*9+3*9+c])-1.0*xvec[(x0*Nx*Nx*Nx+(l+1)*Nx*Nx+k*Nx+j)*3+1]*conj(U[(x0*Nx*Nx*Nx/2+(l+1)*Nx*Nx/2+k*Nx/2+j/2)*8*9+3*9+c+3])-1.0*xvec[(x0*Nx*Nx*Nx+(l+1)*Nx*Nx+k*Nx+j)*3+2]*conj(U[(x0*Nx*Nx*Nx/2+(l+1)*Nx*Nx/2+k*Nx/2+j/2)*8*9+3*9+c+6]);
                            }
                            
                            else {
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+k*Nx+j)*3]*conj(U[(x0*Nx*Nx*Nx/2+k*Nx/2+j/2)*8*9+3*9+c])-1.0*xvec[(x0*Nx*Nx*Nx+k*Nx+j)*3+1]*conj(U[(x0*Nx*Nx*Nx/2+k*Nx/2+j/2)*8*9+3*9+c+3])-1.0*xvec[(x0*Nx*Nx*Nx+k*Nx+j)*3+2]*conj(U[(x0*Nx*Nx*Nx/2+k*Nx/2+j/2)*8*9+3*9+c+6]);
                            }
                            
                        }
                        
                        
                        if((j+k+l)%2==1){
                            if (j-1 >= 0){
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(j-1))*3]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+7*9+3*c]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(j-1))*3+1]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+7*9+3*c+1]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(j-1))*3+2]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+7*9+3*c+2];
                            }
                            else {
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(Nx-1))*3]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+7*9+3*c]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(Nx-1))*3+1]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+7*9+3*c+1]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(Nx-1))*3+2]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+7*9+3*c+2];
                            }
                            if (j+1 < Nx){
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(j+1))*3]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+6*9+3*c]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(j+1))*3+1]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+6*9+3*c+1]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx+(j+1))*3+2]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+6*9+3*c+2];
                            }
                            
                            else {
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx)*3]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+6*9+3*c]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx)*3+1]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+6*9+3*c+1]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+k*Nx)*3+2]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+6*9+3*c+2];
                            }
                            
                            if (k-1 >= 0){
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(k-1)*Nx+j)*3]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+5*9+3*c]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(k-1)*Nx+j)*3+1]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+5*9+3*c+1]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(k-1)*Nx+j)*3+2]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+5*9+3*c+2];
                            }
                            
                            else {
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(Nx-1)*Nx+j)*3]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+5*9+3*c]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(Nx-1)*Nx+j)*3+1]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+5*9+3*c+1]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(Nx-1)*Nx+j)*3+2]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+5*9+3*c+2];
                            }
                            
                            if (k+1 < Nx){
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(k+1)*Nx+j)*3]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+4*9+3*c]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(k+1)*Nx+j)*3+1]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+4*9+3*c+1]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+(k+1)*Nx+j)*3+2]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+4*9+3*c+2];
                                
                            }
                            
                            else {
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+j)*3]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+4*9+3*c]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+j)*3+1]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+4*9+3*c+1]-1.0*xvec[(x0*Nx*Nx*Nx+l*Nx*Nx+j)*3+2]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+4*9+3*c+2];
                            }
                            
                            if (l-1 >= 0) {
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+(l-1)*Nx*Nx+k*Nx+j)*3]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+3*9+3*c]-1.0*xvec[(x0*Nx*Nx*Nx+(l-1)*Nx*Nx+k*Nx+j)*3+1]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+3*9+3*c+1]-1.0*xvec[(x0*Nx*Nx*Nx+(l-1)*Nx*Nx+k*Nx+j)*3+2]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+3*9+3*c+2];
                            }
                            
                            else {
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+(Nx-1)*Nx*Nx+k*Nx+j)*3]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+3*9+3*c]-1.0*xvec[(x0*Nx*Nx*Nx+(Nx-1)*Nx*Nx+k*Nx+j)*3+1]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+3*9+3*c+1]-1.0*xvec[(x0*Nx*Nx*Nx+(Nx-1)*Nx*Nx+k*Nx+j)*3+2]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+3*9+3*c+2];
                            }
                            
                            if (l+1 < Nx){
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+(l+1)*Nx*Nx+k*Nx+j)*3]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+2*9+3*c]-1.0*xvec[(x0*Nx*Nx*Nx+(l+1)*Nx*Nx+k*Nx+j)*3+1]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+2*9+3*c+1]-1.0*xvec[(x0*Nx*Nx*Nx+(l+1)*Nx*Nx+k*Nx+j)*3+2]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+2*9+3*c+2];
                            }
                            
                            else {
                                yvec[row] += -1.0*xvec[(x0*Nx*Nx*Nx+k*Nx+j)*3]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+2*9+3*c]-1.0*xvec[(x0*Nx*Nx*Nx+k*Nx+j)*3+1]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+2*9+3*c+1]-1.0*xvec[(x0*Nx*Nx*Nx+k*Nx+j)*3+2]*U[(x0*Nx*Nx*Nx/2+l*Nx*Nx/2+k*Nx/2+j/2)*8*9+2*9+3*c+2];
                            }
                            
                        }
                    }
                }
            }
        }
    }
    *err = 0;
}

double tracetest(double *evals, int numEvals) {
    double trace;
    int i;
    trace=0;
    for (i=0; i<numEvals; i++) {
        trace+=evals[i];
    }
    return(trace);
}
double trace4test(double *evals, int numEvals) {
    double trace4;
    int i;
    trace4=0;
    for (i=0; i<numEvals; i++) {
        trace4+=pow(evals[i],4);
    }
    return(trace4);
}
