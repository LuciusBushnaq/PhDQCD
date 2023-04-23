/*******************************************************************************
 *
 * File ms6.c
 *
 * Copyright (C) 2017 Nazario Tantalo
 *               2019 Agostino Patella
 *
 * Based on openQCD-1.6/main/ms4.c
 * Copyright (C) 2012, 2013, 2016 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Computation of CA_mu(y0,x0) and CP(y0,x0) correlators.
 *
 * Syntax: ms6 -i <input file> [-noexp] [-a]
 *
 * For the definition of the correlators and for usage instructions see the file
 * README.ms6.
 *
 *changes:
 *
 *Added vector correlator observables V_11,V_12,V_13,V_21 etc.
 *Added distilled  observables V_11_d, CP_d
 *Added the four contributions to distilled observables seperately, V_11_dd (distilled source, distilled sink), V_11_dr (distilled source, rest sink), ...
 *Changed function random_source to return the point on the lattice where our source is placed as an int
 *Changed function random_source to give sources distilled in color and even/odd
 *Created function random_source_cvv to give sources for the vector correlator, distilled in color and even/odd, distilled but identical values in the four spin components
 *Changed function scalar_product to calculate the three contractions of the spinor fields needed for V_ij when given idirac=10,20,30 as argument
 *Changed function correlators to calculate the three contractions of random source vectors eta[0],eta[1],eta[2] needed for V_ij. Real and imaginary parts are both calculated, but only the imaginary component is needed.
 *Changed function correlators to call function slices with argument idirac 10,20,30 and multiply results with respective contraction eta to calculate V_i,j
 *To Implement dilution, changed function correlators to loop over the 12 values of a combined spin color index sc and add up the results for all of them
 *
 *Created function convert_V_source, which transcribes Eigenvector n for timelslice x0 from the vector calculated by primme to a spinor_dble
 *Created function convert_V_sink, which transcribes Eigenvector n for all timelslice from the vector calculated by primme to a spinor_dble
 *Created function copy_spinor, which copies thevalues of the  first argument spinor at timeslice x0 into the second. If x0<0, the whole spinor at every timeslice is copied
 *Created function convert_dist_source, which implements the ket part of applying the distillation operator to a source vector, for a single eigenvector. It takes the scalar product of the source vector with the eigenvector as an argument (real and imaginary part both), as well as the eigenvector itself, and a prefactor for setting the sign
 *Created function convert_dist_sink, which implements the ket part of applying the distillation operator to a solution vector,  for a single eigenvector. It takes the scalar product of the solution vector with the eigenvector as an argument (real and imaginary part both), as well as the eigenvector itself, and a prefactor for setting the sign
 *******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "archive.h"
#include "uflds.h"
#include "u1flds.h"
#include "sflds.h"
#include "linalg.h"
#include "dirac.h"
#include "sap.h"
#include "dfl.h"
#include "forces.h"
#include "version.h"
#include "global.h"

#include <complex.h>
#include "covlasolv.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static struct
{
    int tmax,x0,nsrc;
    int nfl,qhat[2];
    double kappa[2],mu[2];
    double su3csw[2],u1csw[2];
    double cF[2],cF_prime[2];
    double th1[2],th2[2],th3[2];
} file_head;

static struct
{
    int nc;
    double *DV11_re,*DV11_im,*DV11_dist_re,*DV11_dist_im,*DV11_rest_re,*DV11_rest_im,*DV22_re,*DV22_im,*DV22_dist_re,*DV22_dist_im,*DV22_rest_re,*DV22_rest_im,*DV33_re,*DV33_im,*DV33_dist_re,*DV33_dist_im,*DV33_rest_re,*DV33_rest_im,*DVPP_re,*DVPP_im,*DVPP_dist_re,*DVPP_dist_im,*DVPP_rest_re,*DVPP_rest_im, *alpha_DV11_rest_re, *alpha_DV11_rest_im, *alpha_DV22_rest_re, *alpha_DV22_rest_im, *alpha_DV33_rest_re, *alpha_DV33_rest_im, *alpha_DVPP_rest_re, *alpha_DVPP_rest_im;
} data;

#define MAX(n,m) \
if ((n)<(m)) \
(n)=(m)

static int my_rank,noexp,append,endian;
static int first,last,step;
static int ipgrd[2],level,seed;
static int *rlxs_state=NULL,*rlxd_state=NULL;
static double mus;

static int isx_re[L0],isx_im[L0],init=0;
static double corr_re[2*N0],corr_im[2*N0];

static char line[NAME_SIZE];
static char log_dir[NAME_SIZE],dat_dir[NAME_SIZE];
static char loc_dir[NAME_SIZE],cnfg_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE],end_file[NAME_SIZE];
static char par_file[NAME_SIZE],par_save[NAME_SIZE];
static char dat_file[NAME_SIZE],dat_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE],nbase[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fdat=NULL,*fend=NULL;


static void alloc_data(void)
{
    int tmax;
    double *p;
    
    tmax=file_head.tmax;
    
    p=amalloc((24+16+8*20*2)*tmax*sizeof(*p),4);
    error(p==NULL,1,"alloc_data [ms6.c]","Unable to allocate data arrays");
    

    
    data.DV11_re=p;
    data.DV11_im=p+2*tmax;
    data.DV11_dist_re=p+4*tmax;
    data.DV11_dist_im=p+5*tmax;
    data.DV11_rest_re=p+6*tmax;
    data.DV11_rest_im=p+8*tmax;
    data.DV22_re=p+10*tmax;
    data.DV22_im=p+12*tmax;
    data.DV22_dist_re=p+14*tmax;
    data.DV22_dist_im=p+15*tmax;
    data.DV22_rest_re=p+16*tmax;
    data.DV22_rest_im=p+18*tmax;
    data.DV33_re=p+20*tmax;
    data.DV33_im=p+22*tmax;
    data.DV33_dist_re=p+24*tmax;
    data.DV33_dist_im=p+25*tmax;
    data.DV33_rest_re=p+26*tmax;
    data.DV33_rest_im=p+28*tmax;
    data.DVPP_re=p+30*tmax;
    data.DVPP_im=p+32*tmax;
    data.DVPP_dist_re=p+34*tmax;
    data.DVPP_dist_im=p+35*tmax;
    data.DVPP_rest_re=p+36*tmax;
    data.DVPP_rest_im=p+38*tmax;
    data.alpha_DV11_rest_re=p+40*tmax;
    data.alpha_DV11_rest_im=p+80*tmax;
    data.alpha_DV22_rest_re=p+120*tmax;
    data.alpha_DV22_rest_im=p+160*tmax;
    data.alpha_DV33_rest_re=p+200*tmax;
    data.alpha_DV33_rest_im=p+240*tmax;
    data.alpha_DVPP_rest_re=p+280*tmax;
    data.alpha_DVPP_rest_im=p+320*tmax;
    
}


static void write_file_head(void)
{
    int ifl;
    
    write_little_int(1,fdat,4,
                     file_head.tmax,
                     file_head.x0,
                     file_head.nsrc,
                     file_head.nfl);
    
    for (ifl=0;ifl<file_head.nfl;++ifl)
    {
        write_little_int(1,fdat,1,file_head.qhat[ifl]);
        
        write_little_dble(1,fdat,9,
                          file_head.kappa[ifl],
                          file_head.mu[ifl],
                          file_head.su3csw[ifl],
                          file_head.u1csw[ifl],
                          file_head.cF[ifl],
                          file_head.cF_prime[ifl],
                          file_head.th1[ifl],
                          file_head.th2[ifl],
                          file_head.th3[ifl]);
    }
}


static void check_file_head(void)
{
    int ifl;
    
    check_little_int("ms6",fdat,4,
                     file_head.tmax,
                     file_head.x0,
                     file_head.nsrc,
                     file_head.nfl);
    
    for (ifl=0;ifl<file_head.nfl;++ifl)
    {
        check_little_int("ms6",fdat,1,file_head.qhat[ifl]);
        
        check_little_dble("ms6",fdat,9,
                          file_head.kappa[ifl],
                          file_head.mu[ifl],
                          file_head.su3csw[ifl],
                          file_head.u1csw[ifl],
                          file_head.cF[ifl],
                          file_head.cF_prime[ifl],
                          file_head.th1[ifl],
                          file_head.th2[ifl],
                          file_head.th3[ifl]);
    }
}


static void write_data(void)
{
    int tmax;
    
    tmax=file_head.tmax;
    
    write_little_int(1,fdat,1,data.nc);
    write_little_dblearray(1,fdat,tmax*2,data.DV11_re);
    write_little_dblearray(1,fdat,tmax*2,data.DV11_im);
    write_little_dblearray(1,fdat,tmax,data.DV11_dist_re);
    write_little_dblearray(1,fdat,tmax,data.DV11_dist_im);
    write_little_dblearray(1,fdat,tmax*2,data.DV11_rest_re);
    write_little_dblearray(1,fdat,tmax*2,data.DV11_rest_im);
    write_little_dblearray(1,fdat,tmax*2,data.DV22_re);
    write_little_dblearray(1,fdat,tmax*2,data.DV22_im);
    write_little_dblearray(1,fdat,tmax,data.DV22_dist_re);
    write_little_dblearray(1,fdat,tmax,data.DV22_dist_im);
    write_little_dblearray(1,fdat,tmax*2,data.DV22_rest_re);
    write_little_dblearray(1,fdat,tmax*2,data.DV22_rest_im);
    write_little_dblearray(1,fdat,tmax*2,data.DV33_re);
    write_little_dblearray(1,fdat,tmax*2,data.DV33_im);
    write_little_dblearray(1,fdat,tmax,data.DV33_dist_re);
    write_little_dblearray(1,fdat,tmax,data.DV33_dist_im);
    write_little_dblearray(1,fdat,tmax*2,data.DV33_rest_re);
    write_little_dblearray(1,fdat,tmax*2,data.DV33_rest_im);
    write_little_dblearray(1,fdat,tmax*2,data.DVPP_re);
    write_little_dblearray(1,fdat,tmax*2,data.DVPP_im);
    write_little_dblearray(1,fdat,tmax,data.DVPP_dist_re);
    write_little_dblearray(1,fdat,tmax,data.DVPP_dist_im);
    write_little_dblearray(1,fdat,tmax*2,data.DVPP_rest_re);
    write_little_dblearray(1,fdat,tmax*2,data.DVPP_rest_im);
    write_little_dblearray(1,fdat,tmax*40,data.alpha_DV11_rest_re);
    write_little_dblearray(1,fdat,tmax*40,data.alpha_DV11_rest_im);
    write_little_dblearray(1,fdat,tmax*40,data.alpha_DV22_rest_re);
    write_little_dblearray(1,fdat,tmax*40,data.alpha_DV22_rest_im);
    write_little_dblearray(1,fdat,tmax*40,data.alpha_DV33_rest_re);
    write_little_dblearray(1,fdat,tmax*40,data.alpha_DV33_rest_im);
    write_little_dblearray(1,fdat,tmax*40,data.alpha_DVPP_rest_re);
    write_little_dblearray(1,fdat,tmax*40,data.alpha_DVPP_rest_im);
}


static int read_data(void)
{
    int ir,t,tmax;
    stdint_t istd[1];
    double dstd[1];
    
    ir=fread(istd,sizeof(stdint_t),1,fdat);
    
    if (ir!=1)
        return 0;
    
    if (endian==BIG_ENDIAN)
        bswap_int(1,istd);
    
    data.nc=(int)(istd[0]);
    
    tmax=file_head.tmax;
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV11_re[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV11_im[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV11_dist_re[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV11_dist_im[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV11_rest_re[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV11_rest_im[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV22_re[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV22_im[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV22_dist_re[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV22_dist_im[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV22_rest_re[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV22_rest_im[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV33_re[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV33_im[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV33_dist_re[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV33_dist_im[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV33_rest_re[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV33_rest_im[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DVPP_re[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DVPP_im[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV11_dist_re[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV11_dist_im[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV11_rest_re[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.DV11_rest_im[t]=dstd[0];
    }
    error_root(ir!=(1+12*tmax),1,"read_data [ms6.c]",
               "Read error or incomplete data record");
    
    return 1;
}


static void save_data(void)
{
    if (my_rank==0)
    {
        fdat=fopen(dat_file,"ab");
        error_root(fdat==NULL,1,"save_data [ms6.c]",
                   "Unable to open data file");
        write_data();
        fclose(fdat);
    }
}


static void read_dirs(void)
{
    if (my_rank==0)
    {
        find_section("Run name");
        read_line("name","%s",nbase);
        
        find_section("Directories");
        read_line("log_dir","%s",log_dir);
        read_line("dat_dir","%s",dat_dir);
        
        if (noexp)
        {
            read_line("loc_dir","%s",loc_dir);
            cnfg_dir[0]='\0';
        }
        else
        {
            read_line("cnfg_dir","%s",cnfg_dir);
            loc_dir[0]='\0';
        }
        
        find_section("Configurations");
        read_line("first","%d",&first);
        read_line("last","%d",&last);
        read_line("step","%d",&step);
        
        error_root((last<first)||(step<1)||(((last-first)%step)!=0),1,
                   "read_dirs [ms6.c]","Improper configuration range");
        
        find_section("Random number generator");
        read_line("level","%d",&level);
        read_line("seed","%d",&seed);
    }
    
    MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
    
    MPI_Bcast(log_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
    MPI_Bcast(dat_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
    MPI_Bcast(loc_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
    MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
    
    MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);
    
    MPI_Bcast(&level,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);
}


static void setup_files(void)
{
    if (noexp)
        error_root(name_size("%s/%sn%d_%d",loc_dir,nbase,last,NPROC-1)>=NAME_SIZE,
                   1,"setup_files [ms6.c]","loc_dir name is too long");
    else
        error_root(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,
                   1,"setup_files [ms6.c]","cnfg_dir name is too long");
    
    check_dir_root(log_dir);
    check_dir_root(dat_dir);
    error_root(name_size("%s/%s.ms6.log~",log_dir,nbase)>=NAME_SIZE,
               1,"setup_files [ms6.c]","log_dir name is too long");
    error_root(name_size("%s/%s.ms6.dat~",dat_dir,nbase)>=NAME_SIZE,
               1,"setup_files [ms6.c]","dat_dir name is too long");
    
    sprintf(log_file,"%s/%s.ms6.log",log_dir,nbase);
    sprintf(par_file,"%s/%s.ms6.par",dat_dir,nbase);
    sprintf(dat_file,"%s/%s.ms6.dat",dat_dir,nbase);
    sprintf(end_file,"%s/%s.ms6.end",log_dir,nbase);
    sprintf(log_save,"%s~",log_file);
    sprintf(par_save,"%s~",par_file);
    sprintf(dat_save,"%s~",dat_file);
}


static void read_flds_bc_lat_parms(void)
{
    int gg,ifl,bc,cs;
    
    if (my_rank==0)
    {
        find_section("Gauge group");
        read_line("gauge","%s",line);
        
        gg=0;
        if (strcmp(line,"SU(3)")==0)
            gg=1;
        else if (strcmp(line,"U(1)")==0)
            gg=2;
        else if (strcmp(line,"SU(3)xU(1)")==0)
            gg=3;
        else
            error_root(1,1,"read_flds_bc_lat_parms [ms6.c]",
                       "Unknown gauge group %s",line);
        
        find_section("Correlators");
        read_line("nfl","%d",&(file_head.nfl));
        error_root( (file_head.nfl!=1) && (file_head.nfl!=2),1,"read_flds_bc_lat_parms [ms6.c]",
                   "Number of flavours must be 1 or 2 %s",line);
        
    }
    MPI_Bcast(&gg,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&(file_head.nfl),1,MPI_INT,0,MPI_COMM_WORLD);
    
    set_flds_parms(gg,file_head.nfl);
    
    read_bc_parms();
    bc=bc_type();
    cs=bc_cstar();
    
    for (ifl=0;ifl<file_head.nfl;ifl++)
    {
        file_head.qhat[ifl]=0;
        file_head.su3csw[ifl]=0.0;
        file_head.u1csw[ifl]=0.0;
        file_head.cF[ifl]=0.0;
        file_head.cF_prime[ifl]=0.0;
        file_head.th1[ifl]=0.0;
        file_head.th2[ifl]=0.0;
        file_head.th3[ifl]=0.0;
        
        sprintf(line,"Flavour %d",ifl);
        if (my_rank==0)
        {
            find_section(line);
            read_line("kappa","%lf",file_head.kappa+ifl);
            read_line("mu","%lf",file_head.mu+ifl);
            if (gg==1)
            {
                read_line("su3csw","%lf",file_head.su3csw+ifl);
            }
            else if (gg==2)
            {
                read_line("qhat","%d",file_head.qhat+ifl);
                read_line("u1csw","%lf",file_head.u1csw+ifl);
            }
            else if (gg==3)
            {
                read_line("qhat","%d",file_head.qhat+ifl);
                read_line("su3csw","%lf",file_head.su3csw+ifl);
                read_line("u1csw","%lf",file_head.u1csw+ifl);
            }
            if (bc!=3) read_line("cF","%lf",file_head.cF+ifl);
            if (bc==2) read_line("cF'","%lf",file_head.cF_prime+ifl);
            if (cs==0) read_line("theta","%lf %lf %lf",
                                 file_head.th1+ifl,
                                 file_head.th2+ifl,
                                 file_head.th3+ifl);
        }
        
        MPI_Bcast(file_head.qhat+ifl,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(file_head.kappa+ifl,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(file_head.mu+ifl,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(file_head.su3csw+ifl,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(file_head.u1csw+ifl,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(file_head.cF+ifl,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(file_head.cF_prime+ifl,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(file_head.th1+ifl,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(file_head.th2+ifl,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(file_head.th3+ifl,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        
        set_qlat_parms(ifl,
                       file_head.kappa[ifl],
                       file_head.qhat[ifl],
                       file_head.su3csw[ifl],
                       file_head.u1csw[ifl],
                       file_head.cF[ifl],
                       file_head.cF_prime[ifl],
                       file_head.th1[ifl],
                       file_head.th2[ifl],
                       file_head.th3[ifl]);
    }
    
    error( (file_head.nfl==2)&&(file_head.qhat[0]!=file_head.qhat[1]),1,
          "read_flds_bc_lat_parms [ms6.c]",
          "The two flavours must have the same electric charge");
    
    error( (file_head.nfl==2) &&
          (file_head.kappa[0]==file_head.kappa[1]) &&
          (file_head.mu[0]==file_head.mu[1]) &&
          (file_head.qhat[0]==file_head.qhat[1]) &&
          (file_head.su3csw[0]==file_head.su3csw[1]) &&
          (file_head.u1csw[0]==file_head.u1csw[1]) &&
          (file_head.cF[0]==file_head.cF[1]) &&
          (file_head.cF_prime[0]==file_head.cF_prime[1]) &&
          (file_head.th1[0]==file_head.th1[1]) &&
          (file_head.th2[0]==file_head.th2[1]) &&
          (file_head.th3[0]==file_head.th3[1]),1,
          "read_flds_bc_lat_parms [ms6.c]",
          "The two flavours are identical, in this case you have to set nfl=1");
    
    if (my_rank==0)
    {
        find_section("Source fields");
        read_line("x0","%s",&line);
        if (strcmp(line,"random")==0)
            file_head.x0=-1;
        else
            error_root(sscanf(line,"%d",&(file_head.x0))!=1,1,"main [ms6.c]",
                       "x0 must be either 'random' or a non-negative integer");
        read_line("nsrc","%d",&(file_head.nsrc));
        
        error_root( (file_head.x0<-1)||(file_head.x0>=N0),1,"read_fld_bc_lat_parms [ms6.c]",
                   "Specified time x0 is out of range");
        
        /*error_root( (file_head.x0==-1)&&(bc!=3),1,"read_fld_bc_lat_parms [ms6.c]",*/
        /*"Random x0 can be used only with periodic b.c.'s");*/
        
        error_root(((file_head.x0==0)&&(bc!=3))||((file_head.x0==(N0-1))&&(bc==0)),1,
                   "read_bc_parms [ms6.c]","Incompatible choice of boundary "
                   "conditions and source time");
    }
    
    MPI_Bcast(&(file_head.x0),1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&(file_head.nsrc),1,MPI_INT,0,MPI_COMM_WORLD);
    
    file_head.tmax=N0;
    
    if (append)
        check_flds_bc_lat_parms(fdat);
    else
        write_flds_bc_lat_parms(fdat);
}


static void read_solver(void)
{
    solver_parms_t sp;
    
    read_solver_parms(0);
    sp=solver_parms(0);
    
    if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
        read_sap_parms();
    
    if (sp.solver==DFL_SAP_GCR)
    {
        read_dfl_parms(-1);
        read_dfl_parms(sp.idfl);
    }
    
    if (append)
        check_solver_parms(fdat);
    else
        write_solver_parms(fdat);
}


static void read_infile(int argc,char *argv[])
{
    int ifile;
    
    if (my_rank==0)
    {
        flog=freopen("STARTUP_ERROR","w",stdout);
        
        ifile=find_opt(argc,argv,"-i");
        endian=endianness();
        
        error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [ms6.c]",
                   "Syntax: ms6 -i <input file> [-noexp]");
        
        error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [ms6.c]",
                   "Machine has unknown endianness");
        
        noexp=find_opt(argc,argv,"-noexp");
        append=find_opt(argc,argv,"-a");
        
        fin=freopen(argv[ifile+1],"r",stdin);
        error_root(fin==NULL,1,"read_infile [ms6.c]",
                   "Unable to open input file");
    }
    
    MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&noexp,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&append,1,MPI_INT,0,MPI_COMM_WORLD);
    
    read_dirs();
    setup_files();
    
    if (my_rank==0)
    {
        if (append)
            fdat=fopen(par_file,"rb");
        else
            fdat=fopen(par_file,"wb");
        
        error_root(fdat==NULL,1,"read_infile [ms3.c]",
                   "Unable to open parameter file");
    }
    
    read_flds_bc_lat_parms();
    read_solver();
    
    if (my_rank==0)
    {
        fclose(fin);
        fclose(fdat);
        
        if (append==0)
            copy_file(par_file,par_save);
    }
}


static void check_old_log(int *fst,int *lst,int *stp)
{
    int ie,ic,isv;
    int fc,lc,dc,pc;
    int np[4],bp[4];
    
    fend=fopen(log_file,"r");
    error_root(fend==NULL,1,"check_old_log [ms6.c]",
               "Unable to open log file");
    
    fc=0;
    lc=0;
    dc=0;
    pc=0;
    
    ie=0x0;
    ic=0;
    isv=0;
    
    while (fgets(line,NAME_SIZE,fend)!=NULL)
    {
        if (strstr(line,"process grid")!=NULL)
        {
            if (sscanf(line,"%dx%dx%dx%d process grid, %dx%dx%dx%d",
                       np,np+1,np+2,np+3,bp,bp+1,bp+2,bp+3)==8)
            {
                ipgrd[0]=((np[0]!=NPROC0)||(np[1]!=NPROC1)||
                          (np[2]!=NPROC2)||(np[3]!=NPROC3));
                ipgrd[1]=((bp[0]!=NPROC0_BLK)||(bp[1]!=NPROC1_BLK)||
                          (bp[2]!=NPROC2_BLK)||(bp[3]!=NPROC3_BLK));
            }
            else
                ie|=0x1;
        }
        else if (strstr(line,"fully processed")!=NULL)
        {
            pc=lc;
            
            if (sscanf(line,"Configuration no %d",&lc)==1)
            {
                ic+=1;
                isv=1;
            }
            else
                ie|=0x1;
            
            if (ic==1)
                fc=lc;
            else if (ic==2)
                dc=lc-fc;
            else if ((ic>2)&&(lc!=(pc+dc)))
                ie|=0x2;
        }
        else if (strstr(line,"Configuration no")!=NULL)
            isv=0;
    }
    
    fclose(fend);
    
    error_root((ie&0x1)!=0x0,1,"check_old_log [ms6.c]",
               "Incorrect read count");
    error_root((ie&0x2)!=0x0,1,"check_old_log [ms6.c]",
               "Configuration numbers are not equally spaced");
    error_root(isv==0,1,"check_old_log [ms6.c]",
               "Log file extends beyond the last configuration save");
    
    (*fst)=fc;
    (*lst)=lc;
    (*stp)=dc;
}


static void check_old_dat(int fst,int lst,int stp)
{
    int ie,ic;
    int fc,lc,dc,pc;
    
    fdat=fopen(dat_file,"rb");
    error_root(fdat==NULL,1,"check_old_dat [ms6.c]",
               "Unable to open data file");
    
    check_file_head();
    
    fc=0;
    lc=0;
    dc=0;
    pc=0;
    
    ie=0x0;
    ic=0;
    
    while (read_data()==1)
    {
        lc=data.nc;
        ic+=1;
        
        if (ic==1)
            fc=lc;
        else if (ic==2)
            dc=lc-fc;
        else if ((ic>2)&&(lc!=(pc+dc)))
            ie|=0x1;
        
        pc=lc;
    }
    
    fclose(fdat);
    
    error_root(ic==0,1,"check_old_dat [ms6.c]",
               "No data records found");
    error_root((ie&0x1)!=0x0,1,"check_old_dat [ms6.c]",
               "Configuration numbers are not equally spaced");
    error_root((fst!=fc)||(lst!=lc)||(stp!=dc),1,"check_old_dat [ms6.c]",
               "Configuration range is not as reported in the log file");
}


static void check_files(void)
{
    int fst,lst,stp;
    
    ipgrd[0]=0;
    ipgrd[1]=0;
    
    if (my_rank==0)
    {
        if (append)
        {
            check_old_log(&fst,&lst,&stp);
            check_old_dat(fst,lst,stp);
            
            error_root((fst!=lst)&&(stp!=step),1,"check_files [ms6.c]",
                       "Continuation run:\n"
                       "Previous run had a different configuration separation");
            error_root(first!=lst+step,1,"check_files [ms6.c]",
                       "Continuation run:\n"
                       "Configuration range does not continue the previous one");
        }
        else
        {
            fin=fopen(log_file,"r");
            fdat=fopen(dat_file,"rb");
            
            error_root((fin!=NULL)||(fdat!=NULL),1,"check_files [ms6.c]",
                       "Attempt to overwrite old *.log or *.dat file");
            
            fdat=fopen(dat_file,"wb");
            error_root(fdat==NULL,1,"check_files [ms6.c]",
                       "Unable to open data file");
            write_file_head();
            fclose(fdat);
        }
    }
}


static void print_info(void)
{
    int isap,idfl;
    long ip;
    
    if (my_rank==0)
    {
        ip=ftell(flog);
        fclose(flog);
        
        if (ip==0L)
            remove("STARTUP_ERROR");
        
        if (append)
            flog=freopen(log_file,"a",stdout);
        else
            flog=freopen(log_file,"w",stdout);
        
        error_root(flog==NULL,1,"print_info [ms6.c]","Unable to open log file");
        printf("\n");
        
        if (append)
            printf("Continuation run\n\n");
        else
        {
            printf("Computation of CA_mu(y0,x0) and CP(y0,x0) correlators\n");
            printf("-----------------------------------------------------\n\n");
        }
        
        printf("Program version %s\n",openQCD_RELEASE);
        
        if (endian==LITTLE_ENDIAN)
            printf("The machine is little endian\n");
        else
            printf("The machine is big endian\n");
        if (noexp)
            printf("Configurations are read in imported file format\n\n");
        else
            printf("Configurations are read in exported file format\n\n");
        
        if ((ipgrd[0]!=0)&&(ipgrd[1]!=0))
            printf("Process grid and process block size changed:\n");
        else if (ipgrd[0]!=0)
            printf("Process grid changed:\n");
        else if (ipgrd[1]!=0)
            printf("Process block size changed:\n");
        
        if ((append==0)||(ipgrd[0]!=0)||(ipgrd[1]!=0))
        {
            printf("%dx%dx%dx%d lattice, ",N0,N1,N2,N3);
            printf("%dx%dx%dx%d local lattice\n",L0,L1,L2,L3);
            printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
            printf("%dx%dx%dx%d process block size\n\n",
                   NPROC0_BLK,NPROC1_BLK,NPROC2_BLK,NPROC3_BLK);
        }
        
        if (append==0)
        {
            print_bc_parms();
            print_flds_parms();
            print_lat_parms();
            
            printf("Random number generator:\n");
            printf("level = %d, seed = %d\n\n",level,seed);
            
            printf("Source fields:\n");
            if (file_head.x0>=0)
                printf("x0 = %d\n",file_head.x0);
            else
                printf("x0 = random\n");
            printf("nsrc = %d\n\n",file_head.nsrc);
            
            print_solver_parms(&isap,&idfl);
            
            if (isap)
                print_sap_parms(0);
            
            if (idfl)
                print_dfl_parms(0);
        }
        
        printf("Configurations no %d -> %d in steps of %d\n\n",
               first,last,step);
        
        fflush(flog);
    }
}


static void dfl_wsize(int *nws,int *nwv,int *nwvd)
{
    dfl_parms_t dp;
    dfl_pro_parms_t dpp;
    
    dp=dfl_parms();
    dpp=dfl_pro_parms();
    
    MAX(*nws,dp.Ns+2);
    MAX(*nwv,2*dpp.nkv+2);
    MAX(*nwvd,4);
}


static void wsize(int *nws,int *nwsd,int *nwv,int *nwvd)
{
    int nsd;
    solver_parms_t sp;
    
    (*nws)=0;
    (*nwsd)=0;
    (*nwv)=0;
    (*nwvd)=0;
    
    sp=solver_parms(0);
    nsd=2;
    
    if (sp.solver==CGNE)
    {
        MAX(*nws,5);
        MAX(*nwsd,nsd+3);
    }
    else if (sp.solver==SAP_GCR)
    {
        MAX(*nws,2*sp.nkv+1);
        MAX(*nwsd,nsd+2);
    }
    else if (sp.solver==DFL_SAP_GCR)
    {
        MAX(*nws,2*sp.nkv+2);
        MAX(*nwsd,nsd+48);
        dfl_wsize(nws,nwv,nwvd);
    }
    else
        error_root(1,1,"wsize [ms6.c]",
                   "Unknown or unsupported solver");
}


static void random_source_volume(spinor_dble *eta, int co, int e)
{
    int i,iy,ix;
    double r[4];
    complex_dble *c;
    
    set_sd2zero(VOLUME,eta);
    
    
    for (iy=0;iy<(L0*L1*L2*L3);iy++)
    {
        if( ( (e==1) && (iy%2==0) )||( (e==0)&&(iy%2==1) ) ){
            ix=ipt[iy];
            c=(complex_dble*)(eta+ix);
            ranlxd(r,8);
            for(i=0;i<4;++i)
            {
                if (r[2*i]<0.5)
                {
                    c[i*3+co].re=-1.0/sqrt(2.0);
                }
                else
                {
                    c[i*3+co].re=1.0/sqrt(2.0);
                }
                
                if (r[2*i+1]<0.5)
                {
                    c[i*3+co].im=-1.0/sqrt(2.0);
                }
                else
                {
                    c[i*3+co].im=1.0/sqrt(2.0);
                }
            }
            
        }
    }
    
}



static void convert_V_source(spinor_dble **eta,complex_dble **evecs,int x0,int n)
{
    int y0,y1,y2,y3,ix,c;
    complex_dble *c0,*c1,*c2,*c3;
    
    for (c=0;c<4;c++)set_sd2zero(VOLUME,eta[c]);
    y0=x0-cpr[0]*L0;
    
    
    if ((y0>=0)&&(y0<L0)){
        for (y1=0;y1<L1;y1++)
        {
            for (y2=0;y2<L2;y2++)
            {
                for (y3=0;y3<L3;y3++)
                {
                    
                    ix=ipt[y3+y2*L3+y1*L2*L3+y0*L1*L2*L3];
                    c0=(complex_dble*)(eta[0]+ix);
                    c1=(complex_dble*)(eta[1]+ix);
                    c2=(complex_dble*)(eta[2]+ix);
                    c3=(complex_dble*)(eta[3]+ix);
                    for (c=0;c<3;c++)
                    {
                        c0[c].re=evecs[y0][(n*L1*L2*L3+y3+y2*L3+y1*L2*L3)*3+c].re;
                        c0[c].im=evecs[y0][(n*L1*L2*L3+y3+y2*L3+y1*L2*L3)*3+c].im;
                        c1[c+3].re=c0[c].re;
                        c1[c+3].im=c0[c].im;
                        c2[c+6].re=c0[c].re;
                        c2[c+6].im=c0[c].im;
                        c3[c+9].re=c0[c].re;
                        c3[c+9].im=c0[c].im;
                        /*printf("evec number %d real part at %d,%d,%d,%d,%d: %14.6e \n",n,y0,y1,y2,y3,c,evecs[y0][(n*L1*L2*L3+y3+y2*L3+y1*L2*L3)*3+c].re);
                         printf("spinor real part of process %d,%d,%d,%d at %d,%d,%d,%d,%d: %14.6e \n",cpr[0],cpr[1],cpr[2],cpr[3],y0,y1,y2,y3,c,c0[c].re);
                         */
                    }
                }
            }
        }
    }
    
}


/*
static void copy_spinor(spinor_dble *eta,spinor_dble *phi,int x0)
{
    int y0,ix,iy,i;
    complex_dble *c,*d;
    
    set_sd2zero(VOLUME,eta);
    if(x0>=0)
    {
        y0=x0-cpr[0]*L0;
        
        if ((y0>=0)&&(y0<L0))
        {
            for (iy=0;iy<(L1*L2*L3);iy++)
            {
                ix=ipt[iy+y0*L1*L2*L3];
                c=(complex_dble*)(eta+ix);
                d=(complex_dble*)(phi+ix);
                for(i=0;i<12;++i)
                {
                    c[i].re=d[i].re;
                    c[i].im=d[i].im;
                }
            }
            
        }
    }
    else
    {
        for(y0=0;y0<L0;y0++)
        {
            for (iy=0;iy<(L1*L2*L3);iy++)
            {
                ix=ipt[iy+y0*L1*L2*L3];
                c=(complex_dble*)(eta+ix);
                d=(complex_dble*)(phi+ix);
                for(i=0;i<12;++i)
                {
                    c[i].re=d[i].re;
                    c[i].im=d[i].im;
                }
            }
            
        }
    }
}
*/
static void convert_dist_source(spinor_dble *eta,spinor_dble *evec,double rep,double imp,int x0, int pre, double alpha)
{
    int y0,i;
    int ix,iy;
    complex_dble *c1,*c2,temp;
    
    y0=x0-cpr[0]*L0;
    temp.re=rep;
    temp.im=imp;
    if ((y0>=0)&&(y0<L0))
    {
        for (iy=0;iy<(L1*L2*L3);iy++)
        {
            ix=ipt[iy+y0*L1*L2*L3];
            c1=(complex_dble*)(eta+ix);
            c2=(complex_dble*)(evec+ix);
            for(i=0;i<12;++i)
            {
                c1[i].re+=pre*alpha*(c2[i].re*temp.re-c2[i].im*temp.im);
                c1[i].im+=pre*alpha*(c2[i].re*temp.im+c2[i].im*temp.re);
            }
        }
    }
}

static void solve_dirac(spinor_dble *eta,spinor_dble *psi,int *status)
{
    solver_parms_t sp;
    sap_parms_t sap;
    
    sp=solver_parms(0);
    
    if (sp.solver==CGNE)
    {
        mulg5_dble(VOLUME,eta);
        
        tmcg(sp.nmx,sp.res,mus,eta,eta,status);
        
        error_root(status[0]<0,1,"solve_dirac [ms6.c]",
                   "CGNE solver failed (status = %d)",status[0]);
        
        Dw_dble(-mus,eta,psi);
        mulg5_dble(VOLUME,psi);
    }
    else if (sp.solver==SAP_GCR)
    {
        sap=sap_parms();
        set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
        
        sap_gcr(sp.nkv,sp.nmx,sp.res,mus,eta,psi,status);
        
        error_root(status[0]<0,1,"solve_dirac [ms6.c]",
                   "SAP_GCR solver failed (status = %d)",status[0]);
    }
    else if (sp.solver==DFL_SAP_GCR)
    {
        sap=sap_parms();
        set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
        
        dfl_sap_gcr2(sp.idfl,sp.nkv,sp.nmx,sp.res,mus,eta,psi,status);
        
        error_root((status[0]<0)||(status[1]<0),1,
                   "solve_dirac [ms6.c]","DFL_SAP_GCR solver failed "
                   "(status = %d,%d,%d)",status[0],status[1],status[2]);
    }
    else
        error_root(1,1,"solve_dirac [ms6.c]",
                   "Unknown or unsupported solver");
}


static void scalar_product(int idirac,spinor_dble *r,spinor_dble *s, complex_dble *spt)
{
    spt->re=0.0;
    spt->im=0.0;
    
    if (idirac==0)
    {
        spt->re+=(_vector_prod_re((*r).c1,(*s).c1));
        spt->re+=(_vector_prod_re((*r).c2,(*s).c2));
        spt->re+=(_vector_prod_re((*r).c3,(*s).c3));
        spt->re+=(_vector_prod_re((*r).c4,(*s).c4));
        
        spt->im+=(_vector_prod_im((*r).c1,(*s).c1));
        spt->im+=(_vector_prod_im((*r).c2,(*s).c2));
        spt->im+=(_vector_prod_im((*r).c3,(*s).c3));
        spt->im+=(_vector_prod_im((*r).c4,(*s).c4));
    }
    else if (idirac==1)
    {
        spt->re-=(_vector_prod_re((*r).c1,(*s).c3));
        spt->re-=(_vector_prod_re((*r).c2,(*s).c4));
        spt->re-=(_vector_prod_re((*r).c3,(*s).c1));
        spt->re-=(_vector_prod_re((*r).c4,(*s).c2));
        
        spt->im-=(_vector_prod_im((*r).c1,(*s).c3));
        spt->im-=(_vector_prod_im((*r).c2,(*s).c4));
        spt->im-=(_vector_prod_im((*r).c3,(*s).c1));
        spt->im-=(_vector_prod_im((*r).c4,(*s).c2));
    }
    else if (idirac==2)
    {
        spt->re+=(_vector_prod_im((*r).c1,(*s).c4));
        spt->re+=(_vector_prod_im((*r).c2,(*s).c3));
        spt->re-=(_vector_prod_im((*r).c3,(*s).c2));
        spt->re-=(_vector_prod_im((*r).c4,(*s).c1));
        
        spt->im-=(_vector_prod_re((*r).c1,(*s).c4));
        spt->im-=(_vector_prod_re((*r).c2,(*s).c3));
        spt->im+=(_vector_prod_re((*r).c3,(*s).c2));
        spt->im+=(_vector_prod_re((*r).c4,(*s).c1));
    }
    else if (idirac==3)
    {
        spt->re-=(_vector_prod_re((*r).c1,(*s).c4));
        spt->re+=(_vector_prod_re((*r).c2,(*s).c3));
        spt->re+=(_vector_prod_re((*r).c3,(*s).c2));
        spt->re-=(_vector_prod_re((*r).c4,(*s).c1));
        
        spt->im-=(_vector_prod_im((*r).c1,(*s).c4));
        spt->im+=(_vector_prod_im((*r).c2,(*s).c3));
        spt->im+=(_vector_prod_im((*r).c3,(*s).c2));
        spt->im-=(_vector_prod_im((*r).c4,(*s).c1));
    }
    else if (idirac==4)
    {
        spt->re+=(_vector_prod_im((*r).c1,(*s).c3));
        spt->re-=(_vector_prod_im((*r).c2,(*s).c4));
        spt->re-=(_vector_prod_im((*r).c3,(*s).c1));
        spt->re+=(_vector_prod_im((*r).c4,(*s).c2));
        
        spt->im-=(_vector_prod_re((*r).c1,(*s).c3));
        spt->im+=(_vector_prod_re((*r).c2,(*s).c4));
        spt->im+=(_vector_prod_re((*r).c3,(*s).c1));
        spt->im-=(_vector_prod_re((*r).c4,(*s).c2));
    }
    /*for gamma_5 matrix contraction*/
    else if (idirac==5)
    {
        spt->re+=(_vector_prod_re((*r).c1,(*s).c1));
        spt->re+=(_vector_prod_re((*r).c2,(*s).c2));
        spt->re-=(_vector_prod_re((*r).c3,(*s).c3));
        spt->re-=(_vector_prod_re((*r).c4,(*s).c4));
        
        spt->im+=(_vector_prod_im((*r).c1,(*s).c1));
        spt->im+=(_vector_prod_im((*r).c2,(*s).c2));
        spt->im-=(_vector_prod_im((*r).c3,(*s).c3));
        spt->im-=(_vector_prod_im((*r).c4,(*s).c4));
    }
    
    /*for gamma_5 gamma_0 matrix contraction*/
    else if (idirac==10)
    {
        spt->re-=(_vector_prod_re((*r).c1,(*s).c3));
        spt->re-=(_vector_prod_re((*r).c2,(*s).c4));
        spt->re+=(_vector_prod_re((*r).c3,(*s).c1));
        spt->re+=(_vector_prod_re((*r).c4,(*s).c2));
        
        spt->im-=(_vector_prod_im((*r).c1,(*s).c3));
        spt->im-=(_vector_prod_im((*r).c2,(*s).c4));
        spt->im+=(_vector_prod_im((*r).c3,(*s).c1));
        spt->im+=(_vector_prod_im((*r).c4,(*s).c2));
    }
    /*for gamma_5 gamma_1 matrix contraction*/
    else if (idirac==20)
    {
        spt->re+=(_vector_prod_im((*r).c1,(*s).c4));
        spt->re+=(_vector_prod_im((*r).c2,(*s).c3));
        spt->re+=(_vector_prod_im((*r).c3,(*s).c2));
        spt->re+=(_vector_prod_im((*r).c4,(*s).c1));
        
        spt->im-=(_vector_prod_re((*r).c1,(*s).c4));
        spt->im-=(_vector_prod_re((*r).c2,(*s).c3));
        spt->im-=(_vector_prod_re((*r).c3,(*s).c2));
        spt->im-=(_vector_prod_re((*r).c4,(*s).c1));
    }
    /*for gamma_5 gamma_2 matrix contraction*/
    else if (idirac==30)
    {
        spt->re-=(_vector_prod_re((*r).c1,(*s).c4));
        spt->re+=(_vector_prod_re((*r).c2,(*s).c3));
        spt->re-=(_vector_prod_re((*r).c3,(*s).c2));
        spt->re+=(_vector_prod_re((*r).c4,(*s).c1));
        
        spt->im-=(_vector_prod_im((*r).c1,(*s).c4));
        spt->im+=(_vector_prod_im((*r).c2,(*s).c3));
        spt->im-=(_vector_prod_im((*r).c3,(*s).c2));
        spt->im+=(_vector_prod_im((*r).c4,(*s).c1));
    }
    /*for gamma_5 gamma_3 matrix contraction*/
    else if (idirac==40)
    {
        spt->re+=(_vector_prod_im((*r).c1,(*s).c3));
        spt->re-=(_vector_prod_im((*r).c2,(*s).c4));
        spt->re+=(_vector_prod_im((*r).c3,(*s).c1));
        spt->re-=(_vector_prod_im((*r).c4,(*s).c2));
        
        spt->im-=(_vector_prod_re((*r).c1,(*s).c3));
        spt->im+=(_vector_prod_re((*r).c2,(*s).c4));
        spt->im-=(_vector_prod_re((*r).c3,(*s).c1));
        spt->im+=(_vector_prod_re((*r).c4,(*s).c2));
    }
}


void slices(int idirac,spinor_dble *psi1,spinor_dble *psi2)
{
    int bc,ix,t,t0,tmx;
    complex_dble spt;
    
    if (init==0)
    {
        for (t=0;t<L0;t++)
        {
            isx_re[t]=init_hsum(1);
            isx_im[t]=init_hsum(1);
        }
        
        init=1;
    }
    
    bc=bc_type();
    if (bc==0)
        tmx=N0-1;
    else
        tmx=N0;
    t0=cpr[0]*L0;
    
    for (t=0;t<L0;t++)
    {
        reset_hsum(isx_re[t]);
        reset_hsum(isx_im[t]);
    }
    
    for (ix=0;ix<VOLUME;ix++)
    {
        t=global_time(ix);
        
        if (((t>0)&&(t<tmx))||(bc==3))
        {
            t-=t0;
            scalar_product(idirac,psi1+ix,psi2+ix,&spt);
            add_to_hsum(isx_re[t],&(spt.re));
            add_to_hsum(isx_im[t],&(spt.im));
        }
    }
    
    for (t=0;t<2*N0;t++)
    {
        corr_re[t]=0.0;
        corr_im[t]=0.0;
    }
    
    for (t=0;t<L0;t++)
    {
        local_hsum(isx_re[t],&(spt.re));
        local_hsum(isx_im[t],&(spt.im));
        corr_re[t+t0+N0]=spt.re;
        corr_im[t+t0+N0]=spt.im;
    }
    
    if (NPROC>1)
    {
        MPI_Reduce(corr_re+N0,corr_re,N0,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Bcast(corr_re,N0,MPI_DOUBLE,0,MPI_COMM_WORLD);
        
        MPI_Reduce(corr_im+N0,corr_im,N0,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Bcast(corr_im,N0,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    else
    {
        for (t=0;t<N0;t++)
        {
            corr_re[t]=corr_re[t+N0];
            corr_im[t]=corr_im[t+N0];
        }
    }
}



static void correlators(int nc)
{
    int x0,t, i, j;
    int stat[6],idfl,isrc,ifl,ifl2;
    int c1,c2,s1,c,e;
    double wt1,wt2,wt3,wt4,wt[4],wtsum,norm;
    float rn;
    dirac_parms_t dp;
    dfl_parms_t dfl;
    dflst_t dfl_status;
    /*spinors and sources for the distillation sub-space calculation*/
    spinor_dble **wsd_dist_s,**wsd_dist;
    /*Spinors for the stochastic calculation, mormal and rest-space condtioned sources*/
    spinor_dble **wsd_rnd;
    /*complex_dble vector for Primme*/
    complex_dble **evecs;
    /*Parameters for Primme*/
    int dummyv[8];
    double alpha;
    /*Mpi Comm for primme*/
    MPI_Comm primme_comm;
    
    evecs=(complex_dble**)malloc(L0*sizeof(complex_dble*));
    
    if(file_head.nfl==1)
    {
        wsd_dist_s=reserve_wsd(4);
        wsd_dist=reserve_wsd(4);
        
        wsd_rnd=reserve_wsd(3);
        ifl2=1;
        
        
    }
    else
    {
        wsd_dist_s=reserve_wsd(4);
        wsd_dist=reserve_wsd(4);
        
        wsd_rnd=reserve_wsd(4);
        ifl2=2;
    }
    norm=(double)(2.0*file_head.nsrc*N1*N2*N3);
    wtsum=0.0;
    data.nc=nc;
    for (t=0;t<file_head.tmax;++t)
    {
        data.DV11_dist_re[t]=0.0;
        data.DV11_dist_im[t]=0.0;
        data.DV22_dist_re[t]=0.0;
        data.DV22_dist_im[t]=0.0;
        data.DV33_dist_re[t]=0.0;
        data.DV33_dist_im[t]=0.0;
        data.DVPP_dist_re[t]=0.0;
        data.DVPP_dist_im[t]=0.0;
        
        
    }
    
    for(i=0;i<2;++i){
        for (t=0;t<file_head.tmax;++t)
        {
            data.DV11_re[i*file_head.tmax+t]=0.0;
            data.DV11_im[i*file_head.tmax+t]=0.0;
            data.DV11_rest_re[i*file_head.tmax+t]=0.0;
            data.DV11_rest_im[i*file_head.tmax+t]=0.0;
            data.DV22_re[i*file_head.tmax+t]=0.0;
            data.DV22_im[i*file_head.tmax+t]=0.0;
            data.DV22_rest_re[i*file_head.tmax+t]=0.0;
            data.DV22_rest_im[i*file_head.tmax+t]=0.0;
            data.DV33_re[i*file_head.tmax+t]=0.0;
            data.DV33_im[i*file_head.tmax+t]=0.0;
            data.DV33_rest_re[i*file_head.tmax+t]=0.0;
            data.DV33_rest_im[i*file_head.tmax+t]=0.0;
            data.DVPP_re[i*file_head.tmax+t]=0.0;
            data.DVPP_im[i*file_head.tmax+t]=0.0;
            data.DVPP_rest_re[i*file_head.tmax+t]=0.0;
            data.DVPP_rest_im[i*file_head.tmax+t]=0.0;
            for(j=0;j<20;++j){
                data.alpha_DV11_rest_re[i*file_head.tmax*20+j*file_head.tmax+t]=0.0;
                data.alpha_DV11_rest_im[i*file_head.tmax*20+j*file_head.tmax+t]=0.0;
                data.alpha_DV22_rest_re[i*file_head.tmax*20+j*file_head.tmax+t]=0.0;
                data.alpha_DV22_rest_im[i*file_head.tmax*20+j*file_head.tmax+t]=0.0;
                data.alpha_DV33_rest_re[i*file_head.tmax*20+j*file_head.tmax+t]=0.0;
                data.alpha_DV33_rest_im[i*file_head.tmax*20+j*file_head.tmax+t]=0.0;
                data.alpha_DVPP_rest_re[i*file_head.tmax*20+j*file_head.tmax+t]=0.0;
                data.alpha_DVPP_rest_im[i*file_head.tmax*20+j*file_head.tmax+t]=0.0;
                
            }
            
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    wt1=MPI_Wtime();
    dfl=dfl_parms();
    if (dfl.Ns)
    {
        idfl=0;
        while(1)
        {
            dfl_status=dfl_gen_parms(idfl).status;
            if(dfl_status==DFL_OUTOFRANGE) break;
            if(dfl_status==DFL_DEF)
            {
                dfl_modes(idfl,stat);
                error_root(stat[0]<0,1,"main [ms6.c]",
                           "Generation of deflation subspace %d failed (status = %d)",
                           idfl,stat[0]);
                
                if (my_rank==0)
                    printf("Generation of deflation subspace %d: status = %d\n\n",idfl,stat[0]);
            }
            idfl++;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    wt2=MPI_Wtime();
    wt[0]=(wt2-wt1);
    wtsum+=wt[0];
    if (my_rank==0)
    {
        printf("deflation time %.2e sec\n",wt[0]);
        printf("\n");
        fflush(flog);
    }
    /*Get x0. Same x0 for every source, so distillation sub-space doesn't need to be recalculated*/
    if (file_head.x0>=0)
        x0=file_head.x0;
    else
    {
        if (my_rank==0)
        {
            ranlxs(&rn,1);
            x0=(int)(rn*(file_head.tmax-1)+1);
        }
        
        if (NPROC>1)
            MPI_Bcast(&x0,1,MPI_INT,0,MPI_COMM_WORLD);
    }
    /*set distillation multiplier alpha*/
    alpha=0.5;
    /*Distillation subspace calculation*/
    MPI_Barrier(MPI_COMM_WORLD);
    wt1=MPI_Wtime();
    /*Allocate paramaters for Primme*/
    dummyv[0]=NPROC1;
    dummyv[1]=NPROC2;
    dummyv[2]=NPROC3;
    dummyv[3]=L1;
    dummyv[4]=L2;
    dummyv[5]=L3;
    dummyv[6]=0; /*Timeslice*/
    dummyv[7]=0; /*Number of eigenvectors*/
    
    /*Create MPI Comm for Primme*/
    if(dummyv[7]>=0){
        MPI_Comm_split(MPI_COMM_WORLD,cpr[0],cpr[1]*NPROC2*NPROC3+cpr[2]*NPROC3+cpr[3],&primme_comm);
        
        /*Calculate covariant laplacian eigenvectors on every timeslice*/
        for (t=0;t<L0;t++)
        {
            dummyv[6]=t+cpr[0]*L0;
            evecs[t] = (complex_dble*)malloc(dummyv[7]*L1*L2*L3*3*sizeof(complex_dble));
            laplaceeigs(dummyv,cnfg_file,&primme_comm,(complex double*)evecs[t]);
        }
        /*Close MPI Comm for Primme*/
        MPI_Comm_free(&primme_comm);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    wt2=MPI_Wtime();
    wt[0]=(wt2-wt1);
    wtsum+=wt[0];
    
    if (my_rank==0){
        printf("laplacian solver time %.2e sec\n",wt[0]);
        printf("\n");
        fflush(flog);
    }
    
    /*Distillation sub-space disconnected calculation*/
    wt[0]=0.0;
    wt[1]=0.0;
    wt[2]=0.0;
    wt[3]=0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    wt1=MPI_Wtime();
    for (t=0;t<file_head.tmax;t++)
    {
        for(c1=0;c1<dummyv[7];c1++)
        {
            /*Get the eigenvector c1 at source time t*/
            convert_V_source(wsd_dist_s,evecs,t,c1);
            /*Solve dirac equation with eigenvector as source*/
            MPI_Barrier(MPI_COMM_WORLD);
            
            dp=qlat_parms(0);
            set_dirac_parms1(&dp);
            mus=file_head.mu[0];
            
            for(s1=0;s1<4;s1++)solve_dirac(wsd_dist_s[s1],wsd_dist[s1],stat);
            MPI_Barrier(MPI_COMM_WORLD);
            for(s1=0;s1<4;s1++){
                /*Contract dirac solution with complex conjugate of source*/
                slices(2,wsd_dist_s[s1],wsd_dist[s1]);
                data.DV11_dist_re[t]+=corr_re[t]*2.0*file_head.nsrc/norm;
                data.DV11_dist_im[t]+=corr_im[t]*2.0*file_head.nsrc/norm;
                slices(3,wsd_dist_s[s1],wsd_dist[s1]);
                data.DV22_dist_re[t]+=corr_re[t]*2.0*file_head.nsrc/norm;
                data.DV22_dist_im[t]+=corr_im[t]*2.0*file_head.nsrc/norm;
                slices(4,wsd_dist_s[s1],wsd_dist[s1]);
                data.DV33_dist_re[t]+=corr_re[t]*2.0*file_head.nsrc/norm;
                data.DV33_dist_im[t]+=corr_im[t]*2.0*file_head.nsrc/norm;
                slices(5,wsd_dist_s[s1],wsd_dist[s1]);
                data.DVPP_dist_re[t]+=corr_re[t]*2.0*file_head.nsrc/norm;
                data.DVPP_dist_im[t]+=corr_im[t]*2.0*file_head.nsrc/norm;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    wt2=MPI_Wtime();
    wt[0]=(wt2-wt1);
    wtsum+=wt[0];
    if (my_rank==0){
        printf("distilled subspace time %.2e sec\n",wt[0]);
        printf("\n");
        fflush(flog);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    wt1=MPI_Wtime();
    /*Stochastic Disconnected Distillation rest space calculation*/
    for(i=0;i<2;++i){
        for (isrc=0;isrc<file_head.nsrc;isrc++)
        {
            for (e=0;e<2;e++)
            {
                for (c=0;c<3;c++)
                {
                    
                    /*Make stochastic source for disconnected calculation*/
                    MPI_Barrier(MPI_COMM_WORLD);
                    wt3=MPI_Wtime();
                    random_source_volume(wsd_rnd[0],c,e);
                    /*Solve dirac equation for stochastic sources*/
                    for (ifl=0;ifl<file_head.nfl;++ifl)
                    {
                        MPI_Barrier(MPI_COMM_WORLD);
                        dp=qlat_parms(ifl);
                        set_dirac_parms1(&dp);
                        mus=file_head.mu[ifl];
                        solve_dirac(wsd_rnd[0],wsd_rnd[ifl+1],stat+3*ifl);
                        
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    wt4=MPI_Wtime();
                    
                    wt[1]+=wt4-wt3;
                    
                    wt3=MPI_Wtime();
                    /*Set the source*/
                    set_sd2zero(VOLUME,wsd_rnd[2]);
                    /*Apply distillation operator to stochastic sources*/
                    for(c2=0;c2<dummyv[7];c2++){
                        for (t=0;t<file_head.tmax;t++)
                        {
                            /*Get the eigenvector c2 at source time t*/
                            convert_V_source(wsd_dist_s,evecs,t,c2);
                            for(s1=0;s1<4;s1++){
                                /*calculate bra-ket product between eigenvector and pseudoscalar random source*/
                                slices(0,wsd_dist_s[s1],wsd_rnd[0]);
                                /*take bra-ket product result times eigenvector, subtract from rest-space preconditioned sources*/
                                convert_dist_source(wsd_rnd[2],wsd_dist_s[s1],corr_re[t],corr_im[t],t,1.0,1.0);
                            }
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    wt4=MPI_Wtime();
                    wt[2]+=wt4-wt3;
                    
                    wt3=MPI_Wtime();
                    /*Calculate disconnected stochastic observables*/
                    slices(2,wsd_rnd[0],wsd_rnd[ifl2]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DV11_re[i*file_head.tmax+t]+=corr_re[t]/norm;
                        data.DV11_im[i*file_head.tmax+t]+=corr_im[t]/norm;
                        data.DV11_rest_re[i*file_head.tmax+t]+=corr_re[t]/norm;
                        data.DV11_rest_im[i*file_head.tmax+t]+=corr_im[t]/norm;
                        for(j=0;j<20;j++){
                            data.alpha_DV11_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]+=corr_re[t]/norm;
                            data.alpha_DV11_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]+=corr_im[t]/norm;
                        }
                    }
                    slices(2,wsd_rnd[ifl2],wsd_rnd[0]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        
                        data.DV11_re[i*file_head.tmax+t]-=corr_re[t]/norm;
                        data.DV11_im[i*file_head.tmax+t]-=corr_im[t]/norm;
                        data.DV11_rest_re[i*file_head.tmax+t]-=corr_re[t]/norm;
                        data.DV11_rest_im[i*file_head.tmax+t]-=corr_im[t]/norm;
                        for(j=0;j<20;j++){
                            data.alpha_DV11_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]-=corr_re[t]/norm;
                            data.alpha_DV11_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]-=corr_im[t]/norm;
                        }
                    }
                    
                    slices(3,wsd_rnd[0],wsd_rnd[ifl2]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DV22_re[i*file_head.tmax+t]+=corr_re[t]/norm;
                        data.DV22_im[i*file_head.tmax+t]+=corr_im[t]/norm;
                        data.DV22_rest_re[i*file_head.tmax+t]+=corr_re[t]/norm;
                        data.DV22_rest_im[i*file_head.tmax+t]+=corr_im[t]/norm;
                        for(j=0;j<20;j++){
                            data.alpha_DV22_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]+=corr_re[t]/norm;
                            data.alpha_DV22_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]+=corr_im[t]/norm;
                        }
                    }
                    slices(3,wsd_rnd[ifl2],wsd_rnd[0]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DV22_re[i*file_head.tmax+t]-=corr_re[t]/norm;
                        data.DV22_im[i*file_head.tmax+t]-=corr_im[t]/norm;
                        data.DV22_rest_re[i*file_head.tmax+t]-=corr_re[t]/norm;
                        data.DV22_rest_im[i*file_head.tmax+t]-=corr_im[t]/norm;
                        for(j=0;j<20;j++){
                            data.alpha_DV22_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]-=corr_re[t]/norm;
                            data.alpha_DV22_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]-=corr_im[t]/norm;
                        }
                    }
                    
                    slices(4,wsd_rnd[0],wsd_rnd[ifl2]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DV33_re[i*file_head.tmax+t]+=corr_re[t]/norm;
                        data.DV33_im[i*file_head.tmax+t]+=corr_im[t]/norm;
                        data.DV33_rest_re[i*file_head.tmax+t]+=corr_re[t]/norm;
                        data.DV33_rest_im[i*file_head.tmax+t]+=corr_im[t]/norm;
                        for(j=0;j<20;j++){
                            data.alpha_DV33_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]+=corr_re[t]/norm;
                            data.alpha_DV33_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]+=corr_im[t]/norm;
                        }
                    }
                    slices(4,wsd_rnd[ifl2],wsd_rnd[0]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DV33_re[i*file_head.tmax+t]-=corr_re[t]/norm;
                        data.DV33_im[i*file_head.tmax+t]-=corr_im[t]/norm;
                        data.DV33_rest_re[i*file_head.tmax+t]-=corr_re[t]/norm;
                        data.DV33_rest_im[i*file_head.tmax+t]-=corr_im[t]/norm;
                        for(j=0;j<20;j++){
                            data.alpha_DV33_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]-=corr_re[t]/norm;
                            data.alpha_DV33_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]-=corr_im[t]/norm;
                        }
                    }
                    
                    slices(5,wsd_rnd[0],wsd_rnd[ifl2]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DVPP_re[i*file_head.tmax+t]+=corr_re[t]/norm;
                        data.DVPP_im[i*file_head.tmax+t]+=corr_im[t]/norm;
                        data.DVPP_rest_re[i*file_head.tmax+t]+=corr_re[t]/norm;
                        data.DVPP_rest_im[i*file_head.tmax+t]+=corr_im[t]/norm;
                        for(j=0;j<20;j++){
                            data.alpha_DVPP_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]+=corr_re[t]/norm;
                            data.alpha_DVPP_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]+=corr_im[t]/norm;
                        }
                    }
                    slices(5,wsd_rnd[ifl2],wsd_rnd[0]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DVPP_re[i*file_head.tmax+t]+=corr_re[t]/norm;
                        data.DVPP_im[i*file_head.tmax+t]+=corr_im[t]/norm;
                        data.DVPP_rest_re[i*file_head.tmax+t]+=corr_re[t]/norm;
                        data.DVPP_rest_im[i*file_head.tmax+t]+=corr_im[t]/norm;
                        for(j=0;j<20;j++){
                            data.alpha_DVPP_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]+=corr_re[t]/norm;
                            data.alpha_DVPP_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]+=corr_im[t]/norm;
                        }
                    }
                    
                    
                    slices(2,wsd_rnd[2],wsd_rnd[ifl2]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DV11_rest_re[i*file_head.tmax+t]-=corr_re[t]/norm;
                        data.DV11_rest_im[i*file_head.tmax+t]-=corr_im[t]/norm;
                    }
                    for(j=0;j<20;j++){
                        for (t=0;t<file_head.tmax;t++)
                        {
                            data.alpha_DV11_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]-=alpha*corr_re[t]/norm;
                            data.alpha_DV11_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]-=alpha*corr_im[t]/norm;
                        }
                    }
                    slices(2,wsd_rnd[ifl2],wsd_rnd[2]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DV11_rest_re[i*file_head.tmax+t]+=corr_re[t]/norm;
                        data.DV11_rest_im[i*file_head.tmax+t]+=corr_im[t]/norm;
                    }
                    for(j=0;j<20;j++){
                        for (t=0;t<file_head.tmax;t++)
                        {
                            data.alpha_DV11_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]+=alpha*corr_re[t]/norm;
                            data.alpha_DV11_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]+=alpha*corr_im[t]/norm;
                        }
                        
                    }
                    slices(3,wsd_rnd[2],wsd_rnd[ifl2]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DV22_rest_re[i*file_head.tmax+t]-=corr_re[t]/norm;
                        data.DV22_rest_im[i*file_head.tmax+t]-=corr_im[t]/norm;
                    }
                    for(j=0;j<20;j++){
                        for (t=0;t<file_head.tmax;t++)
                        {
                            data.alpha_DV22_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]-=alpha*corr_re[t]/norm;
                            data.alpha_DV22_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]-=alpha*corr_im[t]/norm;
                        }
                    }
                    slices(3,wsd_rnd[ifl2],wsd_rnd[2]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DV22_rest_re[i*file_head.tmax+t]+=corr_re[t]/norm;
                        data.DV22_rest_im[i*file_head.tmax+t]+=corr_im[t]/norm;
                    }
                    for(j=0;j<20;j++){
                        for (t=0;t<file_head.tmax;t++)
                        {
                            data.alpha_DV22_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]+=alpha*corr_re[t]/norm;
                            data.alpha_DV22_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]+=alpha*corr_im[t]/norm;
                        }
                    }
                    slices(4,wsd_rnd[2],wsd_rnd[ifl2]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DV33_rest_re[i*file_head.tmax+t]-=corr_re[t]/norm;
                        data.DV33_rest_im[i*file_head.tmax+t]-=corr_im[t]/norm;
                    }
                    for(j=0;j<20;j++){
                        for (t=0;t<file_head.tmax;t++)
                        {
                            data.alpha_DV33_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]-=alpha*corr_re[t]/norm;
                            data.alpha_DV33_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]-=alpha*corr_im[t]/norm;
                        }
                    }
                    slices(4,wsd_rnd[ifl2],wsd_rnd[2]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DV33_rest_re[i*file_head.tmax+t]+=corr_re[t]/norm;
                        data.DV33_rest_im[i*file_head.tmax+t]+=corr_im[t]/norm;
                    }
                    for(j=0;j<20;j++){
                        for (t=0;t<file_head.tmax;t++)
                        {
                            data.alpha_DV33_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]+=alpha*corr_re[t]/norm;
                            data.alpha_DV33_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]+=alpha*corr_im[t]/norm;
                        }
                    }
                    slices(5,wsd_rnd[2],wsd_rnd[ifl2]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DVPP_rest_re[i*file_head.tmax+t]-=corr_re[t]/norm;
                        data.DVPP_rest_im[i*file_head.tmax+t]-=corr_im[t]/norm;
                    }
                    for(j=0;j<20;j++){
                        for (t=0;t<file_head.tmax;t++)
                        {
                            data.alpha_DVPP_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]-=alpha*corr_re[t]/norm;
                            data.alpha_DVPP_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]-=alpha*corr_im[t]/norm;
                        }
                    }
                    slices(5,wsd_rnd[ifl2],wsd_rnd[2]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        data.DVPP_rest_re[i*file_head.tmax+t]-=corr_re[t]/norm;
                        data.DVPP_rest_im[i*file_head.tmax+t]-=corr_im[t]/norm;
                    }
                    for(j=0;j<20;j++){
                        for (t=0;t<file_head.tmax;t++)
                        {
                            data.alpha_DVPP_rest_re[i*20*file_head.tmax+j*file_head.tmax+t]-=alpha*corr_re[t]/norm;
                            data.alpha_DVPP_rest_im[i*20*file_head.tmax+j*file_head.tmax+t]-=alpha*corr_im[t]/norm;
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    wt4=MPI_Wtime();
                    wt[3]+=wt4-wt3;
                }
            }
        }
    }
    if(file_head.nfl==2)
        wtsum*=0.5;
    MPI_Barrier(MPI_COMM_WORLD);
    wt2=MPI_Wtime();
    wt[0]=(wt2-wt1);
    wtsum+=wt[0];
    if (my_rank==0)
    {
        printf("stochastic subspace Dirac solver time %.2e sec\n",wt[1]);
        printf("\n");
        fflush(flog);
    }
    if (my_rank==0)
    {
        printf("stochastic subspace distillation projection time %.2e sec\n",wt[2]);
        printf("\n");
        fflush(flog);
    }
    if (my_rank==0)
    {
        printf("stochastic subspace solution contraction time %.2e sec\n",wt[3]);
        printf("\n");
        fflush(flog);
    }
    if (my_rank==0)
    {
        printf("stochastic subspace calculation total time %.2e sec\n",wt[0]);
        printf("\n");
        fflush(flog);
    }
    
    release_wsd();
    release_wsd();
    release_wsd();
    
    
    for (t=0;t<L0;t++)
    {
        free(evecs[t]);
    }
    free(evecs);
}


static void save_ranlux(void)
{
    int nlxs,nlxd;
    
    if (rlxs_state==NULL)
    {
        nlxs=rlxs_size();
        nlxd=rlxd_size();
        
        rlxs_state=malloc((nlxs+nlxd)*sizeof(int));
        rlxd_state=rlxs_state+nlxs;
        
        error(rlxs_state==NULL,1,"save_ranlux [ms6.c]",
              "Unable to allocate state arrays");
    }
    
    rlxs_get(rlxs_state);
    rlxd_get(rlxd_state);
}


static void restore_ranlux(void)
{
    rlxs_reset(rlxs_state);
    rlxd_reset(rlxd_state);
}


static void print_log(void)
{
    int t;
    
    if (my_rank==0)
        for (t=0;t<file_head.tmax;++t){
            printf("t = %3d DV_11_re = %14.6e   DV_11_im = %14.6e  DV_11_dist_re = %14.6e  DV_11_dist_im = %14.6e  DV_11_rest_re = %14.6e  DV_11_rest_im = %14.6e   DV_22_re = %14.6e   DV_22_im = %14.6e  DV_22_dist_re = %14.6e  DV_22_dist_im = %14.6e  DV_22_rest_re = %14.6e  DV_22_rest_im = %14.6e   DV_33_re = %14.6e   DV_33_im = %14.6e  DV_33_dist_re = %14.6e  DV_33_dist_im = %14.6e  DV_33_rest_re = %14.6e  DV_33_rest_im = %14.6e   DVPP_re = %14.6e  DVPP_im = %14.6e\n  DV_PP_dist_re = %14.6e  DV_PP_dist_im = %14.6e  DV_PP_rest_re = %14.6e  DV_PP_rest_im = %14.6e\n",
                   t,data.DV11_re[t],data.DV11_im[t],data.DV11_dist_re[t],data.DV11_dist_im[t],data.DV11_rest_re[t],data.DV11_rest_im[t],data.DV22_re[t],data.DV22_im[t],data.DV22_dist_re[t],data.DV22_dist_im[t],data.DV22_rest_re[t],data.DV22_rest_im[t],data.DV33_re[t],data.DV33_im[t],data.DV33_dist_re[t],data.DV33_dist_im[t],data.DV33_rest_re[t],data.DV33_rest_im[t],data.DVPP_re[t],data.DVPP_im[t],data.DVPP_dist_re[t],data.DVPP_dist_im[t],data.DVPP_rest_re[t],data.DVPP_rest_im[t]);
        }
}


static void check_endflag(int *iend)
{
    if (my_rank==0)
    {
        fend=fopen(end_file,"r");
        
        if (fend!=NULL)
        {
            fclose(fend);
            remove(end_file);
            (*iend)=1;
            printf("End flag set, run stopped\n\n");
        }
        else
            (*iend)=0;
    }
    
    MPI_Bcast(iend,1,MPI_INT,0,MPI_COMM_WORLD);
}


int main(int argc,char *argv[])
{
    int nc,iend;
    int nws,nwsd,nwv,nwvd,n;
    double wt1,wt2,wt3,wt4,wtavg;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    read_infile(argc,argv);
    alloc_data();
    check_files();
    print_info();
    
    geometry();
    
    if (append)
        start_ranlux(level,seed^(first-1));
    else
        start_ranlux(level,seed);
    
    wsize(&nws,&nwsd,&nwv,&nwvd);
    alloc_ws(nws);
    alloc_wsd(nwsd);
    alloc_wv(nwv);
    alloc_wvd(nwvd);
    
    iend=0;
    wtavg=0.0;
    
    for (nc=first;(iend==0)&&(nc<=last);nc+=step)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        wt1=MPI_Wtime();
        
        if (my_rank==0){
            printf("Configuration no %d\n\n",nc);
            fflush(flog);
        }
        if (noexp)
        {
            save_ranlux();
            sprintf(cnfg_file,"%s/%sn%d_%d",loc_dir,nbase,nc,my_rank);
            read_cnfg(cnfg_file);
            restore_ranlux();
        }
        else
        {
            sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,nc);
            if (my_rank==0){
                printf("import config else-case start\n");
                fflush(flog);
            }
            import_cnfg(cnfg_file);
            if (my_rank==0){
                printf("import config else-case end\n");
                fflush(flog);
            }
            udfld();
            set_flags(UPDATED_UD);
            adfld();
            set_flags(UPDATED_AD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        wt3=MPI_Wtime();
        correlators(nc);
        MPI_Barrier(MPI_COMM_WORLD);
        wt4=MPI_Wtime();
        save_data();
        print_log();
        
        MPI_Barrier(MPI_COMM_WORLD);
        wt2=MPI_Wtime();
        wtavg+=(wt2-wt1);
        
        if (my_rank==0)
        {
            n=(nc-first)/step+1;
            
            printf("Configuration no %d fully processed in %.2e sec ",
                   nc,wt2-wt1);
            printf("(average = %.2e sec)\n\n",wtavg/(double)(n));
            printf("correlator function time = %.2e sec\n\n",wt4-wt3);
            
            fflush(flog);
            copy_file(log_file,log_save);
            copy_file(dat_file,dat_save);
        }
        
        check_endflag(&iend);
    }
    
    if (my_rank==0)
    {
        fflush(flog);
        copy_file(log_file,log_save);
        fclose(flog);
    }
    
    MPI_Finalize();
    exit(0);
}
