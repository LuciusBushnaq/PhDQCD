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


double tracetest(double *evals, int numEvals);
double trace4test(double *evals, int numEvals);

int laplaceeigs (const int dummy[8], char *configname, MPI_Comm *comm, complex double* evecs);
