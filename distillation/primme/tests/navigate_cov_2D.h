#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
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
    complex double*                buffer[4];      /* MPI buffers*/
}
Field;

Field alloc_field()
{
    int i;
    Field phi;
    
    phi.data = NULL;
    
    /* North and South buffers: length = 3*nxl*nzl */
    phi.buffer[0] = malloc(sizeof(complex double)*nxl);
    for (i=0;i<nxl;i++) phi.buffer[0][i] = 0;
    phi.buffer[1] = malloc(sizeof(complex double)*nxl);
    for (i=0;i<nxl;i++) phi.buffer[1][i] = 0;
    /* East  and West  buffers: length = 3*nyl*nzl */
    phi.buffer[2] = malloc(sizeof(complex double)*nyl);
    for (i=0;i<nyl;i++) phi.buffer[2][i] = 0;
    phi.buffer[3] = malloc(sizeof(complex double)*nyl);
    for (i=0;i<nyl;i++) phi.buffer[3][i] = 0;
    
    return phi;
}

void free_field(Field phi)
{
    int i;
    for (i=0;i<4;i++) free(phi.buffer[i]);
}

complex double neigh(Field* phi, int site, int hop)
{
    /* Get (x,y,z)*/
    static const int hop_x[6] = {0, 0,-1,+1, 0, 0};
    static const int hop_y[6] = {-1,+1, 0, 0, 0,0};
    static const int hop_z[6] = {0, 0, 0, 0,-1,+1};
    
    int y = (site)/nxl + hop_y[hop];
    int x = site%nxl + hop_x[hop];
    
    
    if (y==-1 ) return phi->buffer[0][x];
    if (y==nyl) return phi->buffer[1][x];
    if (x==-1 ) return phi->buffer[2][y];
    if (x==nxl) return phi->buffer[3][y];
    
    return phi->data[x+nxl*y];

}


