#include <stdlib.h>
#include <stdio.h>

// Buffers...
#define SOUTH 0         //              1
#define NORTH 1         //      2       #       3
#define WEST  2         //              0
#define EAST  3         //

// Struct to hold Field and MPI buffers
typedef struct
{
  // int                        nxl;            // Local array size - x-extent
  // int                        nyl;            // Local array size - y-extent
  float*                data;           // bulk data
  float*                buffer[4];      // MPI buffers
}
Field;

Field alloc_field()
{
  int i;
  Field phi;
  
  // phi.nxl = nxl;
  // phi.nyl = nyl;
  phi.data = NULL;

// North and South buffers: length = nxl
  phi.buffer[0] = malloc(sizeof(float) * nxl);
  for (i=0;i<nxl;i++) phi.buffer[0][i] = 0;
  phi.buffer[1] = malloc(sizeof(float) * nxl);
  for (i=0;i<nxl;i++) phi.buffer[1][i] = 0;
// East  and West  buffers: length = nyl
  phi.buffer[2] = malloc(sizeof(float) * nyl);
  for (i=0;i<nyl;i++) phi.buffer[2][i] = 0;
  phi.buffer[3] = malloc(sizeof(float) * nyl);
  for (i=0;i<nyl;i++) phi.buffer[3][i] = 0;
  
  return phi;
}

void free_field(Field phi)
{
  int i;
  for (i=0;i<4;i++) free(phi.buffer[i]);
}

float neigh(Field* phi, int site, int hop)
{
// Get (x,y)
  static const int hop_x[4] = { 0, 0,-1,+1};
  static const int hop_y[4] = {-1,+1, 0, 0};

  int x = site % nxl + hop_x[hop];
  int y = site / nxl + hop_y[hop];

  if (y==-1 ) return phi->buffer[0][x];
  if (y==nyl) return phi->buffer[1][x];
  if (x==-1 ) return phi->buffer[2][y];
  if (x==nxl) return phi->buffer[3][y];

  return phi->data[x+nxl*y];

  return 0.0;
}




#if 0

int main()
{
  int i;
  int nx=4,ny=3;
  float* data = malloc(sizeof(float)*nx*ny);
  
  Field phi = alloc_field(nx,ny,data);
    for (i=0;i<nx*ny;i++)
      data[i] = 100.0 * (float) i;

    for (i=0;i<nx*ny;i++)
      printf("%lf\n", neigh(&phi, i, 3));

    free_field(phi);
    free(data);
    return 0;
  }
#endif
