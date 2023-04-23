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
    complex double*                buffer[6];      /* MPI buffers*/
}
Field;

Field alloc_field()
{
    int i;
    Field phi;
    
    phi.data = NULL;
    
    /* North and South buffers: length = 3*nxl*nzl */
    phi.buffer[0] = malloc(sizeof(complex double)*nxl*nzl);
    for (i=0;i<nxl*nzl;i++) phi.buffer[0][i] = 0;
    phi.buffer[1] = malloc(sizeof(complex double)*nxl*nzl);
    for (i=0;i<nxl*nzl;i++) phi.buffer[1][i] = 0;
    /* East  and West  buffers: length = 3*nyl*nzl */
    phi.buffer[2] = malloc(sizeof(complex double)*nyl*nzl);
    for (i=0;i<nyl*nzl;i++) phi.buffer[2][i] = 0;
    phi.buffer[3] = malloc(sizeof(complex double)*nyl*nzl);
    for (i=0;i<nyl*nzl;i++) phi.buffer[3][i] = 0;
    /* Up  and   Down  buffers: length = 3*nxl*nyl */
    phi.buffer[4] = malloc(sizeof(complex double)*nxl*nyl);
    for (i=0;i<nxl*nyl;i++) phi.buffer[4][i] = 0;
    phi.buffer[5] = malloc(sizeof(complex double)*nxl*nyl);
    for (i=0;i<nxl*nyl;i++) phi.buffer[5][i] = 0;
    
    return phi;
}

void free_field(Field phi)
{
    int i;
    for (i=0;i<6;i++) free(phi.buffer[i]);
}

complex double neigh(Field* phi, int site, int hop)
{
    /* Get (x,y,z)*/
    static const int hop_x[6] = {0, 0,-1,+1, 0, 0};
    static const int hop_y[6] = {-1,+1, 0, 0, 0,0};
    static const int hop_z[6] = {0, 0, 0, 0,-1,+1};
    
    int z = site / (nxl*nyl);
    int y = (site-nxl*nyl*z)/nxl;
    int x = site%nxl;
    z+=hop_z[hop];
    y+=hop_y[hop];
    x+=hop_x[hop];
    
    
    if (y==-1 ) return phi->buffer[0][x+nxl*z];
    if (y==nyl) return phi->buffer[1][x+nxl*z];
    if (x==-1 ) return phi->buffer[2][y+nyl*z];
    if (x==nxl) return phi->buffer[3][y+nyl*z];
    if (z==-1 ) return phi->buffer[4][x+nxl*y];
    if (z==nzl) return phi->buffer[5][x+nxl*y];
    
    return phi->data[x+nxl*y+nxl*nyl*z];

}


