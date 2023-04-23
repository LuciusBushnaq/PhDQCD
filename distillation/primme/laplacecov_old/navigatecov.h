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
    phi.buffer[0] = malloc(sizeof(complex double)*3*nxl*nzl);
    for (i=0;i<3*nxl*nzl;i++) phi.buffer[0][i] = 0;
    phi.buffer[1] = malloc(sizeof(complex double)*3*nxl*nzl);
    for (i=0;i<3*nxl*nzl;i++) phi.buffer[1][i] = 0;
    /* East  and West  buffers: length = 3*nyl*nzl */
    phi.buffer[2] = malloc(sizeof(complex double)*3*nyl*nzl);
    for (i=0;i<3*nyl*nzl;i++) phi.buffer[2][i] = 0;
    phi.buffer[3] = malloc(sizeof(complex double)*3*nyl*nzl);
    for (i=0;i<3*nyl*nzl;i++) phi.buffer[3][i] = 0;
    /* Up  and   Down  buffers: length = 3*nxl*nyl */
    phi.buffer[4] = malloc(sizeof(complex double)*3*nxl*nyl);
    for (i=0;i<3*nxl*nyl;i++) phi.buffer[4][i] = 0;
    phi.buffer[5] = malloc(sizeof(complex double)*3*nxl*nyl);
    for (i=0;i<3*nxl*nyl;i++) phi.buffer[5][i] = 0;
    
    return phi;
}

void free_field(Field phi)
{
    int i;
    for (i=0;i<6;i++) free(phi.buffer[i]);
}

complex double neigh(Field* phi, int site, int hop, int comp)
{
    /* Get (x,y,z)*/
    static const int hop_x[6] = {0, 0,-1,+1, 0, 0};
    static const int hop_y[6] = {-1,+1, 0, 0, 0,0};
    static const int hop_z[6] = {0, 0, 0, 0,-1,+1};
    
    int z = site / (3*nxl*nyl);
    int y = (site-3*nxl*nyl*z)/(3*nxl);
    int x = (site-3*nxl*nyl*z-3*nxl*y) / 3;
    int c = site%3;
    z+=hop_z[hop];
    y+=hop_y[hop];
    x+=hop_x[hop];
    
    
    if (y==-1 ) return phi->buffer[0][comp+3*x+3*nxl*z];
    if (y==nyl) return phi->buffer[1][comp+3*x+3*nxl*z];
    if (x==-1 ) return phi->buffer[2][comp+3*y+3*nyl*z];
    if (x==nxl) return phi->buffer[3][comp+3*y+3*nyl*z];
    if (z==-1 ) return phi->buffer[4][comp+3*x+3*nxl*y];
    if (z==nzl) return phi->buffer[5][comp+3*x+3*nxl*y];
    
    return phi->data[comp+3*x+3*nxl*y+3*nxl*nyl*z];

}

complex double config(complex double* U, int site, int hop, int rank, int comp)
{
    /* Get global (x,y,z)*/
    int ngx=npx*nxl;
    int ngy=npy*nyl;
    int ngz=npz*nzl;
    
    int zl = site / (3*nxl*nyl);
    int yl = (site-3*nxl*nyl*zl)/(3*nxl);
    int xl = (site-3*nxl*nyl*zl-3*nxl*yl) / 3;
    int  c = site%3;
    
    int pz=rank/(npx*npy);
    int py = (rank-npx*npy*pz)/npx;
    int px = rank % npx;
    
    int x=px*nxl+xl;
    int y=py*nyl+yl;
    int z=pz*nzl+zl;

    if((x+y+z)%2==0){
        if(hop==0){
            if(y>0) return conj(U[(z*ngy*ngx/2+(y-1)*ngx/2+x/2)*8*9+4*9+c+3*comp]);
            else return conj(U[(z*ngy*ngx/2+(ngy-1)*ngx/2+x/2)*8*9+4*9+c+3*comp]);
        }
        if(hop==1){
            if(y<ngy-1) return conj(U[(z*ngy*ngx/2+(y+1)*ngx/2+x/2)*8*9+5*9+c+3*comp]);
            else return conj(U[(z*ngy*ngx/2+x/2)*8*9+5*9+c+3*comp]);
        }
        if(hop==2){
            if(x>0) return conj(U[(z*ngy*ngx/2+y*ngx/2+(x-1)/2)*8*9+6*9+c+3*comp]);
            else return conj(U[(z*ngy*ngx/2+y*ngx/2+(ngx-1)/2)*8*9+6*9+c+3*comp]);
        }
        if(hop==3){
            if(x<ngx-1) return conj(U[(z*ngy*ngx/2+y*ngx/2+(x+1)/2)*8*9+7*9+c+3*comp]);
            else return conj(U[(z*ngy*ngx/2+y*ngx/2)*8*9+7*9+c+3*comp]);
            
        }
        if(hop==4){
            if(z>0) return conj(U[((z-1)*ngy*ngx/2+y*ngx/2+x/2)*8*9+2*9+c+3*comp]);
            else return conj(U[((ngz-1)*ngy*ngx/2+y*ngx/2+x/2)*8*9+2*9+c+3*comp]);
        }
        if(hop==5){
            if(z<ngz-1) return conj(U[((z+1)*ngy*ngx/2+y*ngx/2+x/2)*8*9+3*9+c+3*comp]);
            else return conj(U[(y*ngx/2+x/2)*8*9+3*9+c+3*comp]);
        }
    }
    if((x+y+z)%2==1){
        if(hop==0) return U[(z*ngy*ngx/2+y*ngx/2+x/2)*8*9+5*9+3*c+comp];
        
        if(hop==1) return U[(z*ngy*ngx/2+y*ngx/2+x/2)*8*9+4*9+3*c+comp];
       
        if(hop==2) return U[(z*ngy*ngx/2+y*ngx/2+x/2)*8*9+7*9+3*c+comp];
      
        if(hop==3) return U[(z*ngy*ngx/2+y*ngx/2+x/2)*8*9+6*9+3*c+comp];
            
        
        if(hop==4) return U[(z*ngy*ngx/2+y*ngx/2+x/2)*8*9+3*9+3*c+comp];
        
        if(hop==5) return U[(z*ngy*ngx/2+y*ngx/2+x/2)*8*9+2*9+3*c+comp];
        
    }
    
}

void importconfig(const char * configname){
    int x;
    int y;
    int z;
    int mu;
    int cc;
    /* get global lattice sizes */
    int nxg=npx*nxl;
    int nyg=npy*nyl;
    int nzg=npz*nzl;
    
    FILE *infile;
    char charblock[16];
    infile = fopen(configname,"r");
    for (z=0; z<nzg; z++) {
        for (y=0; y<nyg; y++) {
            for (x=0; x<nxg/2; x++) {
                for (mu=0; mu<8; mu++) {
                    for (cc=0; cc<9; cc++) {
                        fseek(infile, 24+(x0*nzg*nyg*nxg/2+z*nyg*nxg/2+y*nxg/2+x)*8*9*16+mu*9*16+cc*16, SEEK_SET);
                        fread(charblock, 16, 1, infile);
                        U[(z*nyg*nxg/2+y*nxg/2+x)*8*9+mu*9+cc]=*(complex double*)charblock;
                    }
                }
            }
        }
    }
    fclose(infile);
    
    
}
