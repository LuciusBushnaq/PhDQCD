 
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
    double *P,*P_d,*P_dd,*P_rr,*P_dr,*P_rd,*P_t,*A0,*A1,*A2,*A3,*V00,*V00_d,*V00_dd,*V00_rr,*V00_dr,*V00_rd,*V01,*V02,*V03,*V03_d,*V03_dd,*V03_rr,*V03_dr,*V03_rd,*V10,*V11,*V11_d,*V11_dd,*V11_rr,*V11_dr,*V11_rd,*V12,*V13,*V20,*V21,*V22,*V22_d,*V22_dd,*V22_rr,*V22_dr,*V22_rd,*V23,*V30,*V31,*V32,*V33,*V33_d,*V33_dd,*V33_rr,*V33_dr,*V33_rd;
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
    
    p=amalloc(52*tmax*sizeof(*p),4);
    error(p==NULL,1,"alloc_data [ms6.c]","Unable to allocate data arrays");
    
    data.P=p;
    data.P_d=p+1*tmax;;
    data.P_dd=p+2*tmax;
    data.P_rr=p+3*tmax;
    data.P_dr=p+4*tmax;
    data.P_rd=p+5*tmax;
    data.P_t=p+6*tmax;
    data.A0=p+7*tmax;
    data.A1=p+8*tmax;
    data.A2=p+9*tmax;
    data.A3=p+10*tmax;
    data.V00=p+11*tmax;
    data.V00_d=p+12*tmax;
    data.V00_dd=p+13*tmax;
    data.V00_rr=p+14*tmax;
    data.V00_dr=p+15*tmax;
    data.V00_rd=p+16*tmax;
    data.V01=p+17*tmax;
    data.V02=p+18*tmax;
    data.V03=p+19*tmax;
    data.V03_d=p+20*tmax;
    data.V03_dd=p+21*tmax;
    data.V03_rr=p+22*tmax;
    data.V03_dr=p+23*tmax;
    data.V03_rd=p+24*tmax;
    data.V10=p+25*tmax;
    data.V11=p+26*tmax;
    data.V11_d=p+27*tmax;
    data.V11_dd=p+28*tmax;
    data.V11_rr=p+29*tmax;
    data.V11_dr=p+30*tmax;
    data.V11_rd=p+31*tmax;
    data.V12=p+32*tmax;
    data.V13=p+33*tmax;
    data.V20=p+34*tmax;
    data.V21=p+35*tmax;
    data.V22=p+36*tmax;
    data.V22_d=p+37*tmax;
    data.V22_dd=p+38*tmax;
    data.V22_rr=p+39*tmax;
    data.V22_dr=p+40*tmax;
    data.V22_rd=p+41*tmax;
    data.V23=p+42*tmax;
    data.V30=p+43*tmax;
    data.V31=p+44*tmax;
    data.V32=p+45*tmax;
    data.V33=p+46*tmax;
    data.V33_d=p+47*tmax;
    data.V33_dd=p+48*tmax;
    data.V33_rr=p+49*tmax;
    data.V33_dr=p+50*tmax;
    data.V33_rd=p+51*tmax;
    
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
    
    write_little_dblearray(1,fdat,tmax,data.P);
    write_little_dblearray(1,fdat,tmax,data.P_d);
    write_little_dblearray(1,fdat,tmax,data.P_dd);
    write_little_dblearray(1,fdat,tmax,data.P_rr);
    write_little_dblearray(1,fdat,tmax,data.P_dr);
    write_little_dblearray(1,fdat,tmax,data.P_rd);
    write_little_dblearray(1,fdat,tmax,data.P_t);
    write_little_dblearray(1,fdat,tmax,data.A0);
    write_little_dblearray(1,fdat,tmax,data.A1);
    write_little_dblearray(1,fdat,tmax,data.A2);
    write_little_dblearray(1,fdat,tmax,data.A3);
    write_little_dblearray(1,fdat,tmax,data.V00);
    write_little_dblearray(1,fdat,tmax,data.V00_d);
    write_little_dblearray(1,fdat,tmax,data.V00_dd);
    write_little_dblearray(1,fdat,tmax,data.V00_rr);
    write_little_dblearray(1,fdat,tmax,data.V00_dr);
    write_little_dblearray(1,fdat,tmax,data.V00_rd);
    write_little_dblearray(1,fdat,tmax,data.V01);
    write_little_dblearray(1,fdat,tmax,data.V02);
    write_little_dblearray(1,fdat,tmax,data.V03);
    write_little_dblearray(1,fdat,tmax,data.V03_d);
    write_little_dblearray(1,fdat,tmax,data.V03_dd);
    write_little_dblearray(1,fdat,tmax,data.V03_rr);
    write_little_dblearray(1,fdat,tmax,data.V03_dr);
    write_little_dblearray(1,fdat,tmax,data.V03_rd);
    write_little_dblearray(1,fdat,tmax,data.V10);
    write_little_dblearray(1,fdat,tmax,data.V11);
    write_little_dblearray(1,fdat,tmax,data.V11_d);
    write_little_dblearray(1,fdat,tmax,data.V11_dd);
    write_little_dblearray(1,fdat,tmax,data.V11_rr);
    write_little_dblearray(1,fdat,tmax,data.V11_dr);
    write_little_dblearray(1,fdat,tmax,data.V11_rd);
    write_little_dblearray(1,fdat,tmax,data.V12);
    write_little_dblearray(1,fdat,tmax,data.V13);
    write_little_dblearray(1,fdat,tmax,data.V20);
    write_little_dblearray(1,fdat,tmax,data.V21);
    write_little_dblearray(1,fdat,tmax,data.V22);
    write_little_dblearray(1,fdat,tmax,data.V22_d);
    write_little_dblearray(1,fdat,tmax,data.V22_dd);
    write_little_dblearray(1,fdat,tmax,data.V22_rr);
    write_little_dblearray(1,fdat,tmax,data.V22_dr);
    write_little_dblearray(1,fdat,tmax,data.V22_rd);
    write_little_dblearray(1,fdat,tmax,data.V23);
    write_little_dblearray(1,fdat,tmax,data.V30);
    write_little_dblearray(1,fdat,tmax,data.V31);
    write_little_dblearray(1,fdat,tmax,data.V32);
    write_little_dblearray(1,fdat,tmax,data.V33);
    write_little_dblearray(1,fdat,tmax,data.V33_d);
    write_little_dblearray(1,fdat,tmax,data.V33_dd);
    write_little_dblearray(1,fdat,tmax,data.V33_rr);
    write_little_dblearray(1,fdat,tmax,data.V33_dr);
    write_little_dblearray(1,fdat,tmax,data.V33_rd);
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
        
        data.P[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.P_d[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.P_dd[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.P_rr[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.P_dr[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.P_rd[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.P_t[t]=dstd[0];
    }
    
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.A0[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.A1[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.A2[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.A3[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V00[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V00_dd[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V00_rr[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V00_dr[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V00_rd[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V01[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V02[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V03[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V03_d[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V03_dd[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V03_rr[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V03_dr[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V03_rd[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V10[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V11[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V11_d[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V11_dd[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V11_rr[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V11_dr[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V11_rd[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V12[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V13[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V20[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V21[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V22[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V22_d[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V22_dd[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V22_rr[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V22_dr[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V22_rd[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V23[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V30[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V31[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V32[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V33[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V33_d[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V33_dd[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V33_rr[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V33_dr[t]=dstd[0];
    }
    for (t=0;t<tmax;t++)
    {
        ir+=fread(dstd,sizeof(double),1,fdat);
        
        if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
        
        data.V33_rd[t]=dstd[0];
    }
    error_root(ir!=(1+53*tmax),1,"read_data [ms6.c]",
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
        MAX(*nwsd,nsd+190);
        dfl_wsize(nws,nwv,nwvd);
    }
    else
        error_root(1,1,"wsize [ms6.c]",
                   "Unknown or unsupported solver");
}

static void random_source(int x0,spinor_dble *eta)
{
    int y0,iy,ix,i;
    double twopi, r[12];
    complex_dble *c;
    
    twopi=8.0*atan(1.0);
    set_sd2zero(VOLUME,eta);
    y0=x0-cpr[0]*L0;
    
    
    if ((y0>=0)&&(y0<L0)){
        for (iy=0;iy<(L1*L2*L3);iy++)
        {
            ix=ipt[iy+y0*L1*L2*L3];
            c=(complex_dble*)(eta+ix);
            ranlxd(r,12);
            for (i=0;i<12;i++)
            {
                c[i].re=cos(twopi*r[i]);
                c[i].im=sin(twopi*r[i]);
            }
        }
    }
}
static void random_source_cvv(int x0,spinor_dble **eta)
{
    int y0,iy,ix;
    int c;
    double val,r[6];
    complex_dble *c0,*c1,*c2,*c3;
    
    val=1.0/sqrt(2.0);
    
    set_sd2zero(VOLUME,eta[0]);
    set_sd2zero(VOLUME,eta[1]);
    set_sd2zero(VOLUME,eta[2]);
    set_sd2zero(VOLUME,eta[3]);
    y0=x0-cpr[0]*L0;
    
    if ((y0>=0)&&(y0<L0)){
        for (iy=0;iy<(L1*L2*L3);iy++)
        {
            ix=ipt[iy+y0*L1*L2*L3];
            c0=(complex_dble*)(eta[0]+ix);
            c1=(complex_dble*)(eta[1]+ix);
            c2=(complex_dble*)(eta[2]+ix);
            c3=(complex_dble*)(eta[3]+ix);
            ranlxd(r,6);
            for(c=0;c<3;c++){
                c0[c].re=(r[2*c]>0.5)?-val:val;
                c0[c].im=(r[2*c+1]>0.5)?-val:val;
                c1[c+3].re=c0[c].re;
                c1[c+3].im=c0[c].im;
                c2[c+6].re=c0[c].re;
                c2[c+6].im=c0[c].im;
                c3[c+9].re=c0[c].re;
                c3[c+9].im=c0[c].im;
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

static void convert_V_sink(spinor_dble **eta,complex_dble **evecs,int n)
{
    int y0,y1,y2,y3,ix,c;
    complex_dble *c0,*c1,*c2,*c3;
    
    /*
     printf("convert_V_sink started with n=%d \n",n);
     fflush(flog);
     */
    for (c=0;c<4;c++)set_sd2zero(VOLUME,eta[c]);
    
    for (y0=0;y0<L0;y0++)
    {
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
                    }
                }
            }
        }
    }
    /*
     printf("convert_V_sink ended with n=%d \n",n);
     fflush(flog);
     */
}
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
static void convert_dist_sink(spinor_dble *eta,spinor_dble *evec,double *rep,double *imp, int pre, double alpha)
{
    int y0,i;
    int ix,iy;
    complex_dble *c1,*c2,temp;
    
    
    for(y0=0;y0<L0;y0++)
    {
        temp.re=rep[y0+cpr[0]*L0];
        temp.im=imp[y0+cpr[0]*L0];
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
    int x0,t,tt;
    int stat[6],idfl,isrc,ifl,ifl2;
    int c1,c2,s1,s2,s3,s4,j;
    double wt1,wt2,wt[4],wtsum,norm;
    float rn;
    dirac_parms_t dp;
    dfl_parms_t dfl;
    dflst_t dfl_status;
    /*spinors and sources for the distillation sub-space calculation*/
    spinor_dble **wsd_dist_s,**wsd_dist;
    /*Spinors and sources for the stochastic calculation, mormal, distilled-space conditioned, rest-space condtioned*/
    spinor_dble **wsd_rnd, **wsd_rnd_dist, **wsd_rnd_rest;
    spinor_dble **wsd_rnd_cvv, **wsd_rnd_cvv_dist, **wsd_rnd_cvv_rest;
    spinor_dble **wsd_rnd_cvv_s, **wsd_rnd_cvv_s_dist, **wsd_rnd_cvv_s_rest;
    /*complex_dble vector for Primme*/
    complex_dble **evecs;
    /*Parameters for Primme*/
    int dummyv[8];
    double alpha;
    /*Mpi Comm for primme*/
    MPI_Comm primme_comm;
    /*gamma_i*gamma_1 matrices for Vector-Vector 11 component in distillation sub-space*/
    const double gamma5_0[4][4]= {{0, 0,-1.0, 0},{0, 0, 0,-1.0},{-1.0,0, 0, 0},{0 ,-1.0, 0, 0}};
    const double gamma5_1[4][4]= {{0, 0, 0,-1.0},{0, 0,-1.0, 0},{0,-1.0, 0, 0},{-1.0, 0, 0, 0}}; /*Omitted i*/
    const double gamma5_2[4][4]= {{0, 0, 0,-1.0},{0, 0,+1.0, 0},{0,-1.0, 0, 0},{+1.0, 0, 0, 0}};
    const double gamma5_3[4][4]= {{0, 0,-1.0, 0},{0, 0, 0,+1.0},{-1.0,0, 0, 0},{0 ,+1.0, 0, 0}}; /*Omitted i*/
    /*Temp variables to hold contraction results for calculation in distillation sub-space*/
    complex_dble C[4][4][64];
    

    /*C[4][4]=(complex_dble*)malloc(N0*sizeof(complex_dble));*/
    evecs=(complex_dble**)malloc(L0*sizeof(complex_dble*));
    
    if(file_head.nfl==1)
    {
        wsd_dist_s=reserve_wsd(4);
        wsd_dist=reserve_wsd(4);
        
        wsd_rnd=reserve_wsd(2);
        wsd_rnd_dist=reserve_wsd(2);
        wsd_rnd_rest=reserve_wsd(2);
        
        wsd_rnd_cvv=reserve_wsd(4);
        wsd_rnd_cvv_dist=reserve_wsd(4);
        wsd_rnd_cvv_rest=reserve_wsd(4);
        wsd_rnd_cvv_s=reserve_wsd(4);
        wsd_rnd_cvv_s_dist=reserve_wsd(4);
        wsd_rnd_cvv_s_rest=reserve_wsd(4);
        
        ifl2=1;
        
        
    }
    else
    {
        wsd_dist_s=reserve_wsd(4);
        wsd_dist=reserve_wsd(4);
        
        wsd_rnd=reserve_wsd(3);
        wsd_rnd_dist=reserve_wsd(3);
        wsd_rnd_rest=reserve_wsd(3);
        
        wsd_rnd_cvv=reserve_wsd(4);
        wsd_rnd_cvv_dist=reserve_wsd(4);
        wsd_rnd_cvv_rest=reserve_wsd(4);
        wsd_rnd_cvv_s=reserve_wsd(4);
        wsd_rnd_cvv_s_dist=reserve_wsd(4);
        wsd_rnd_cvv_s_rest=reserve_wsd(4);

        ifl2=2;
    }
    norm=(double)(file_head.nsrc*N1*N2*N3);
    wtsum=0.0;
    data.nc=nc;
    for (t=0;t<file_head.tmax;++t)
    {
        data.P[t]=0.0;
        data.P_d[t]=0.0;
        data.P_dd[t]=0.0;
        data.P_rr[t]=0.0;
        data.P_dr[t]=0.0;
        data.P_rd[t]=0.0;
        data.P_t[t]=0.0;
        data.A0[t]=0.0;
        data.A1[t]=0.0;
        data.A2[t]=0.0;
        data.A3[t]=0.0;
        data.V00[t]=0.0;
        data.V00_d[t]=0.0;
        data.V00_dd[t]=0.0;
        data.V00_rr[t]=0.0;
        data.V00_dr[t]=0.0;
        data.V00_rd[t]=0.0;
        data.V01[t]=0.0;
        data.V02[t]=0.0;
        data.V03[t]=0.0;
        data.V03_d[t]=0.0;
        data.V03_dd[t]=0.0;
        data.V03_rr[t]=0.0;
        data.V03_dr[t]=0.0;
        data.V03_rd[t]=0.0;
        data.V10[t]=0.0;
        data.V11[t]=0.0;
        data.V11_d[t]=0.0;
        data.V11_dd[t]=0.0;
        data.V11_rr[t]=0.0;
        data.V11_dr[t]=0.0;
        data.V11_rd[t]=0.0;
        data.V12[t]=0.0;
        data.V13[t]=0.0;
        data.V20[t]=0.0;
        data.V21[t]=0.0;
        data.V22[t]=0.0;
        data.V22_d[t]=0.0;
        data.V22_dd[t]=0.0;
        data.V22_rr[t]=0.0;
        data.V22_dr[t]=0.0;
        data.V22_rd[t]=0.0;
        data.V23[t]=0.0;
        data.V30[t]=0.0;
        data.V31[t]=0.0;
        data.V32[t]=0.0;
        data.V33[t]=0.0;
        data.V33_d[t]=0.0;
        data.V33_dd[t]=0.0;
        data.V33_rr[t]=0.0;
        data.V33_dr[t]=0.0;
        data.V33_rd[t]=0.0;
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
    alpha=8.5;
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
    dummyv[7]=20; /*Number of eigenvectors*/
    
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
    
    /*Distillation sub-space calculation*/
    wt[0]=0.0;
    wt[1]=0.0;
    wt[2]=0.0;
    wt[3]=0.0;
    for(c1=0;c1<dummyv[7];c1++){
        /*Get the eigenvector c2 at source time x0*/
        MPI_Barrier(MPI_COMM_WORLD);
        wt1=MPI_Wtime();
        
        convert_V_source(wsd_dist_s,evecs,x0,c1);
        
        wt2=MPI_Wtime();
        wt[0]+=(wt2-wt1);
        /*Solve dirac equation with eigenvector as source*/
        MPI_Barrier(MPI_COMM_WORLD);
        wt1=MPI_Wtime();
        
        dp=qlat_parms(0);
        set_dirac_parms1(&dp);
        mus=file_head.mu[0];
        for(s1=0;s1<4;s1++)solve_dirac(wsd_dist_s[s1],wsd_dist[s1],stat+3);
        MPI_Barrier(MPI_COMM_WORLD);
        wt2=MPI_Wtime();
        wt[1]+=(wt2-wt1);
        
        MPI_Barrier(MPI_COMM_WORLD);
        wt1=MPI_Wtime();
        for(c2=0;c2<dummyv[7];c2++){
            /*Get the eigenvector c2 at all times*/
            convert_V_sink(wsd_dist_s,evecs,c2);
            for(s1=0;s1<4;s1++){
                for(s2=0;s2<4;s2++){
                    /*Contract dirac solution with complex conjugate for pseudoscalar observable in distillation subspace*/
                    slices(0,wsd_dist_s[s1],wsd_dist[s2]);
                    for (t=0;t<file_head.tmax;t++)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        data.P_dd[tt]+=(corr_re[t]*corr_re[t]+corr_im[t]*corr_im[t])*pow(alpha,2.0)*file_head.nsrc/norm;
                        /*Save contraction results for use in vector-vector */
                        C[s1][s2][t].re=corr_re[t];
                        C[s1][s2][t].im=corr_im[t];
                    }
                }
            }
            /*Form vector-vector 11 observable from C and gamma_5*gamma_1 matrix*/
            for (t=0;t<file_head.tmax;t++)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                for(s1=0;s1<4;s1++){
                    for(s2=0;s2<4;s2++){
                        for(s3=0;s3<4;s3++){
                            for(s4=0;s4<4;s4++){
                                /*minus signs from commuting one gamma_5 matrix, and from the omitted i*i for  V11_dd,V33_dd*/
                                data.V00_dd[tt]-=((C[s1][s2][t].re*C[s3][s4][t].re+C[s1][s2][t].im*C[s3][s4][t].im)*gamma5_0[s3][s1]*gamma5_0[s2][s4])*pow(alpha,2.0)*file_head.nsrc/norm;
                                data.V11_dd[tt]+=((C[s1][s2][t].re*C[s3][s4][t].re+C[s1][s2][t].im*C[s3][s4][t].im)*gamma5_1[s3][s1]*gamma5_1[s2][s4])*pow(alpha,2.0)*file_head.nsrc/norm;
                                data.V22_dd[tt]-=((C[s1][s2][t].re*C[s3][s4][t].re+C[s1][s2][t].im*C[s3][s4][t].im)*gamma5_2[s3][s1]*gamma5_2[s2][s4])*pow(alpha,2.0)*file_head.nsrc/norm;
                                data.V33_dd[tt]+=((C[s1][s2][t].re*C[s3][s4][t].re+C[s1][s2][t].im*C[s3][s4][t].im)*gamma5_3[s3][s1]*gamma5_3[s2][s4])*pow(alpha,2.0)*file_head.nsrc/norm;
                                
                                data.V03_dd[tt]+=((C[s1][s2][t].re*C[s3][s4][t].re+C[s1][s2][t].im*C[s3][s4][t].im)*gamma5_0[s3][s1]*gamma5_3[s2][s4])*pow(alpha,2.0)*file_head.nsrc/norm;
                                data.P_t[tt]-=((C[s1][s2][t].re*C[s3][s4][t].im-C[s1][s2][t].im*C[s3][s4][t].re)*gamma5_1[s3][s1]*gamma5_1[s2][s4])*pow(alpha,2.0)*file_head.nsrc/norm;
                            }
                        }
                    }
                }
            }
            
        }
        MPI_Barrier(MPI_COMM_WORLD);
        wt2=MPI_Wtime();
        wt[2]+=(wt2-wt1);
        
    }
    wtsum+=wt[0]+wt[1]+wt[2];
    if (my_rank==0){
        printf("distillation sub-space preprocessing time %.2e sec\n",wt[0]);
        printf("\n");
        printf("distillation sub-space Dirac solver time %.2e sec\n",wt[1]);
        printf("\n");
        printf("distillation sub-space postprocessing time %.2e sec\n",wt[2]);
        printf("\n");
        fflush(flog);
    }
    
    
    
    /*Stochastic Distillation rest space calculation*/
    for (isrc=0;isrc<file_head.nsrc;isrc++)
    {
        wt[0]=0.0;
        wt[1]=0.0;
        wt[2]=0.0;
        wt[3]=0.0;
        
        MPI_Barrier(MPI_COMM_WORLD);
        wt1=MPI_Wtime();
        /*Make stochastic sources for Pseudoscalar and Vector-Vector*/
        random_source(x0,wsd_rnd[0]);
        random_source_cvv(x0,wsd_rnd_cvv_s);
        /*Copy the stochastic sources into the rest-space preconditioned sources, set distilled space preconditioned sources to zero*/
        copy_spinor(wsd_rnd_rest[0],wsd_rnd[0],x0);
        set_sd2zero(VOLUME,wsd_rnd_dist[0]);
        for(s1=0;s1<4;s1++)copy_spinor(wsd_rnd_cvv_s_rest[s1],wsd_rnd_cvv_s[s1],x0);
        for(s1=0;s1<4;s1++)set_sd2zero(VOLUME,wsd_rnd_cvv_s_dist[s1]);
        
        /*Apply distillation operator to stochastic sources, add/subtract result for distilled/rest-space preconditioned sources*/
        for(c2=0;c2<dummyv[7];c2++){
            /*Get the eigenvector c2 at source time x0*/
            convert_V_source(wsd_dist_s,evecs,x0,c2);
            for(s1=0;s1<4;s1++){
                /*calculate bra-ket product between eigenvector and pseudoscalar random source*/
                slices(0,wsd_dist_s[s1],wsd_rnd[0]);
                /*take bra-ket product result times eigenvector, add to distilled space preconditioned sources, subtract from rest-space preconditioned sources*/
                convert_dist_source(wsd_rnd_dist[0],wsd_dist_s[s1],corr_re[x0],corr_im[x0],x0,1,alpha);
                convert_dist_source(wsd_rnd_rest[0],wsd_dist_s[s1],corr_re[x0],corr_im[x0],x0,-1,alpha);
                
                /*same again for vector-vector sources*/
                slices(0,wsd_dist_s[s1],wsd_rnd_cvv_s[s1]);
                convert_dist_source(wsd_rnd_cvv_s_dist[s1],wsd_dist_s[s1],corr_re[x0],corr_im[x0],x0,1,alpha);
                convert_dist_source(wsd_rnd_cvv_s_rest[s1],wsd_dist_s[s1],corr_re[x0],corr_im[x0],x0,-1,alpha);
            }
            
        }
        MPI_Barrier(MPI_COMM_WORLD);
        wt2=MPI_Wtime();
        wt[0]+=(wt2-wt1);
        /*Solve dirac equation for stochastic source, rest/distilled-space preconditioned sources, for pseudoscalar, and vector-vector correlator*/
        
        for (ifl=0;ifl<file_head.nfl;++ifl)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            wt1=MPI_Wtime();
            dp=qlat_parms(ifl);
            set_dirac_parms1(&dp);
            mus=file_head.mu[ifl];
            /*Pseudoscalar*/
            solve_dirac(wsd_rnd[0],wsd_rnd[ifl+1],stat+3*ifl);
            solve_dirac(wsd_rnd_rest[0],wsd_rnd_rest[ifl+1],stat+3*ifl);
            solve_dirac(wsd_rnd_dist[0],wsd_rnd_dist[ifl+1],stat+3*ifl);
            /*Vector-Vector*/
            for(s1=0;s1<4;s1++) solve_dirac(wsd_rnd_cvv_s[s1],wsd_rnd_cvv[s1],stat+3*ifl);
            for(s1=0;s1<4;s1++) solve_dirac(wsd_rnd_cvv_s_rest[s1],wsd_rnd_cvv_rest[s1],stat+3*ifl);
            for(s1=0;s1<4;s1++) solve_dirac(wsd_rnd_cvv_s_dist[s1],wsd_rnd_cvv_dist[s1],stat+3*ifl);
            MPI_Barrier(MPI_COMM_WORLD);
            wt2=MPI_Wtime();
            wt[1]+=(wt2-wt1);
        }
            
        
        if (my_rank==0)
        {
            printf("Source n. %d, x0=%d, preprocessing time=%.2e\n",isrc,x0,wt[0]);
            printf("Source n. %d, x0=%d, computation of propagators completed\n",isrc,x0);
            
            for (ifl=0;ifl<file_head.nfl;++ifl)
            {
                printf("propagator_%d: ",ifl+1);
                if (dfl.Ns)
                {
                    printf("status = %d,%d",stat[0+3*ifl],stat[1+3*ifl]);
                    
                    if (stat[2+3*ifl])
                        printf(" (no of subspace regenerations = %d) ",stat[2+3*ifl]);
                    else
                        printf(" ");
                }
                else
                    printf("status = %d ",stat[0+3*ifl]);
                printf("time %.2e sec\n",wt[1]);
            }
            printf("\n");
            fflush(flog);
        }
        /*Apply rest-space post-conditioning to the stochastic dirac solutions*/
        MPI_Barrier(MPI_COMM_WORLD);
        wt1=MPI_Wtime();
        /*Copy solutions to temp spinors, so they're not lost in subsequent step*/
        copy_spinor(wsd_rnd_rest[0],wsd_rnd_rest[1],-1);
        copy_spinor(wsd_rnd_dist[0],wsd_rnd_dist[1],-1);
        for(s1=0;s1<4;s1++) copy_spinor(wsd_rnd_cvv_s_rest[s1],wsd_rnd_cvv_rest[s1],-1);
        for(s1=0;s1<4;s1++) copy_spinor(wsd_rnd_cvv_s_dist[s1],wsd_rnd_cvv_dist[s1],-1);
        
        /*Apply distillation operator to dirac solutions, subtract result to get rest-space postconditioned solutions*/
        for(c2=0;c2<dummyv[7];c2++){
            /*Get eigenvector c2 at all timeslices*/
            convert_V_sink(wsd_dist_s,evecs,c2);
            for(s1=0;s1<4;s1++){
                /*calculate bra-ket product between eigenvector and pseudoscalar rest-preconditioned dirac solutions*/
                slices(0,wsd_dist_s[s1],wsd_rnd_rest[0]);
                /*take bra-ket product result times eigenvector, subtract for rest-space postconditioned, rest-preconditioned  solutions*/
                convert_dist_sink(wsd_rnd_rest[1],wsd_dist_s[s1],corr_re,corr_im,-1,alpha);
                
                /*calculate bra-ket product between eigenvector and pseudoscalar distilled-preconditioned dirac solutions*/
                slices(0,wsd_dist_s[s1],wsd_rnd_dist[0]);
                /*take bra-ket product result times eigenvector, subtract for rest-space postconditioned, distilled-preconditioned  solutions*/
                convert_dist_sink(wsd_rnd_dist[1],wsd_dist_s[s1],corr_re,corr_im,-1,alpha);
                
                /*Same for vector-vector. Solutions and eigenvectors of different spin no longer orthogonal, thus double sum*/
                for(s2=0;s2<4;s2++){
                    slices(0,wsd_dist_s[s1],wsd_rnd_cvv_s_rest[s2]);
                    convert_dist_sink(wsd_rnd_cvv_rest[s2],wsd_dist_s[s1],corr_re,corr_im,-1,alpha);
                    slices(0,wsd_dist_s[s1],wsd_rnd_cvv_s_dist[s2]);
                    convert_dist_sink(wsd_rnd_cvv_dist[s2],wsd_dist_s[s1],corr_re,corr_im,-1,alpha);
                }
                
            }
        }
            
        
        MPI_Barrier(MPI_COMM_WORLD);
        wt2=MPI_Wtime();
        wt[2]+=(wt2-wt1);
        /*Calculate rest-space post-conditioned stochastic observables*/
        MPI_Barrier(MPI_COMM_WORLD);
        wt1=MPI_Wtime();
                
                slices(0,wsd_rnd_rest[1],wsd_rnd[ifl2]);
                for (t=0;t<file_head.tmax;++t)
                {
                    tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                    data.P_rr[tt]+=corr_re[t]/norm;
                }
                slices(0,wsd_rnd_dist[1],wsd_rnd[ifl2]);
                for (t=0;t<file_head.tmax;++t)
                {
                    tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                    data.P_dr[tt]+=corr_re[t]/norm;
                }
                /* V_0j */
                
                for(j=1;j<5;j++){
                    /* V_0j */
                    slices(j*10,wsd_rnd_cvv_rest[0],wsd_rnd_cvv[2]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==1) data.V00_rr[tt]+=corr_re[t]/norm;
                        if(j==4) data.V03_rr[tt]+=corr_re[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_rest[1],wsd_rnd_cvv[3]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==1) data.V00_rr[tt]+=corr_re[t]/norm;
                        if(j==4) data.V03_rr[tt]+=corr_re[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_rest[2],wsd_rnd_cvv[0]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==1) data.V00_rr[tt]-=corr_re[t]/norm;
                        if(j==4) data.V03_rr[tt]-=corr_re[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_rest[3],wsd_rnd_cvv[1]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==1) data.V00_rr[tt]-=corr_re[t]/norm;
                        if(j==4) data.V03_rr[tt]-=corr_re[t]/norm;
                    }
                    
                    /* V_1j */
                    slices(j*10,wsd_rnd_cvv_rest[0],wsd_rnd_cvv[3]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==2) data.V11_rr[tt]-=corr_im[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_rest[1],wsd_rnd_cvv[2]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==2) data.V11_rr[tt]-=corr_im[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_rest[2],wsd_rnd_cvv[1]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==2) data.V11_rr[tt]-=corr_im[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_rest[3],wsd_rnd_cvv[0]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==2) data.V11_rr[tt]-=corr_im[t]/norm;
                    }
                    
                    /* V_2j */
                    slices(j*10,wsd_rnd_cvv_rest[0],wsd_rnd_cvv[3]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==3) data.V22_rr[tt]+=corr_re[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_rest[1],wsd_rnd_cvv[2]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==3) data.V22_rr[tt]-=corr_re[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_rest[2],wsd_rnd_cvv[1]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==3) data.V22_rr[tt]+=corr_re[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_rest[3],wsd_rnd_cvv[0]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==3) data.V22_rr[tt]-=corr_re[t]/norm;
                    }
                    /* V_3j */
                    slices(j*10,wsd_rnd_cvv_rest[0],wsd_rnd_cvv[2]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==4) data.V33_rr[tt]-=corr_im[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_rest[1],wsd_rnd_cvv[3]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==4) data.V33_rr[tt]+=corr_im[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_rest[2],wsd_rnd_cvv[0]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==4) data.V33_rr[tt]-=corr_im[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_rest[3],wsd_rnd_cvv[1]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==4) data.V33_rr[tt]+=corr_im[t]/norm;
                    }
                }
                for(j=1;j<5;j++){
                    /* V_0j */
                    slices(j*10,wsd_rnd_cvv_dist[0],wsd_rnd_cvv[2]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==1) data.V00_dr[tt]+=corr_re[t]/norm;
                        if(j==4) data.V03_dr[tt]+=corr_re[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_dist[1],wsd_rnd_cvv[3]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==1) data.V00_dr[tt]+=corr_re[t]/norm;
                        if(j==4) data.V03_dr[tt]+=corr_re[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_dist[2],wsd_rnd_cvv[0]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==1) data.V00_dr[tt]-=corr_re[t]/norm;
                        if(j==4) data.V03_dr[tt]-=corr_re[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_dist[3],wsd_rnd_cvv[1]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==1) data.V00_dr[tt]-=corr_re[t]/norm;
                        if(j==4) data.V03_dr[tt]-=corr_re[t]/norm;
                    }
                    
                    /* V_1j */
                    slices(j*10,wsd_rnd_cvv_dist[0],wsd_rnd_cvv[3]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==2) data.V11_dr[tt]-=corr_im[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_dist[1],wsd_rnd_cvv[2]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==2) data.V11_dr[tt]-=corr_im[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_dist[2],wsd_rnd_cvv[1]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==2) data.V11_dr[tt]-=corr_im[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_dist[3],wsd_rnd_cvv[0]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==2) data.V11_dr[tt]-=corr_im[t]/norm;
                    }
                    
                    /* V_2j */
                    slices(j*10,wsd_rnd_cvv_dist[0],wsd_rnd_cvv[3]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==3) data.V22_dr[tt]+=corr_re[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_dist[1],wsd_rnd_cvv[2]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==3) data.V22_dr[tt]-=corr_re[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_dist[2],wsd_rnd_cvv[1]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==3) data.V22_dr[tt]+=corr_re[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_dist[3],wsd_rnd_cvv[0]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==3) data.V22_dr[tt]-=corr_re[t]/norm;
                    }
                    /* V_3j */
                    slices(j*10,wsd_rnd_cvv_dist[0],wsd_rnd_cvv[2]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==4) data.V33_dr[tt]-=corr_im[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_dist[1],wsd_rnd_cvv[3]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==4) data.V33_dr[tt]+=corr_im[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_dist[2],wsd_rnd_cvv[0]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==4) data.V33_dr[tt]-=corr_im[t]/norm;
                    }
                    slices(j*10,wsd_rnd_cvv_dist[3],wsd_rnd_cvv[1]);
                    for (t=0;t<file_head.tmax;++t)
                    {
                        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                        if(j==4) data.V33_dr[tt]+=corr_im[t]/norm;
                    }
                }
                
        MPI_Barrier(MPI_COMM_WORLD);
        wt2=MPI_Wtime();
        wt[3]+=(wt2-wt1);
        /*Apply distilled-space post-conditioning to the stochastic dirac solutions*/
        MPI_Barrier(MPI_COMM_WORLD);
        wt1=MPI_Wtime();
        /*Set the spinors from previous calculation to zero again*/
        set_sd2zero(VOLUME,wsd_rnd_rest[1]);
        for(s1=0;s1<4;s1++) set_sd2zero(VOLUME,wsd_rnd_cvv_rest[s1]);
        for(c2=0;c2<dummyv[7];c2++){
            /*Get eigenvector c2 at all timeslices*/
            convert_V_sink(wsd_dist_s,evecs,c2);
            for(s1=0;s1<4;s1++){
                /*calculate bra-ket product between eigenvector and pseudoscalar rest-preconditioned dirac solutions*/
                slices(0,wsd_dist_s[s1],wsd_rnd_rest[0]);
                /*take bra-ket product result times eigenvector, add for distilled-space postconditioned, rest-preconditioned  solutions*/
                convert_dist_sink(wsd_rnd_rest[1],wsd_dist_s[s1],corr_re,corr_im,1,alpha);
                
                /*Same for vector-vector. Solutions and eigenvectors of different spin no longer orthogonal, thus double sum*/
                for(s2=0;s2<4;s2++){
                    slices(0,wsd_dist_s[s1],wsd_rnd_cvv_s_rest[s2]);
                    convert_dist_sink(wsd_rnd_cvv_rest[s2],wsd_dist_s[s1],corr_re,corr_im,1,alpha);
                }
            }
            
        }
        MPI_Barrier(MPI_COMM_WORLD);
        wt2=MPI_Wtime();
        wt[2]+=(wt2-wt1);
        /*Calculate distilled-space post-conditioned stochastic observables*/
        MPI_Barrier(MPI_COMM_WORLD);
        wt1=MPI_Wtime();
        
        slices(0,wsd_rnd_rest[1],wsd_rnd[ifl2]);
        for (t=0;t<file_head.tmax;++t)
        {
            tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
            data.P_rd[tt]+=corr_re[t]/norm;
        }
        
        for(j=1;j<5;j++){
            /* V_0j */
            slices(j*10,wsd_rnd_cvv_rest[0],wsd_rnd_cvv[2]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V00_rd[tt]+=corr_re[t]/norm;
                if(j==4) data.V03_rd[tt]+=corr_re[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv_rest[1],wsd_rnd_cvv[3]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V00_rd[tt]+=corr_re[t]/norm;
                if(j==4) data.V03_rd[tt]+=corr_re[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv_rest[2],wsd_rnd_cvv[0]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V00_rd[tt]-=corr_re[t]/norm;
                if(j==4) data.V03_rd[tt]-=corr_re[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv_rest[3],wsd_rnd_cvv[1]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V00_rd[tt]-=corr_re[t]/norm;
                if(j==4) data.V03_rd[tt]-=corr_re[t]/norm;
            }
            
            /* V_1j */
            slices(j*10,wsd_rnd_cvv_rest[0],wsd_rnd_cvv[3]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==2) data.V11_rd[tt]-=corr_im[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv_rest[1],wsd_rnd_cvv[2]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==2) data.V11_rd[tt]-=corr_im[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv_rest[2],wsd_rnd_cvv[1]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==2) data.V11_rd[tt]-=corr_im[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv_rest[3],wsd_rnd_cvv[0]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==2) data.V11_rd[tt]-=corr_im[t]/norm;
            }
            
            /* V_2j */
            slices(j*10,wsd_rnd_cvv_rest[0],wsd_rnd_cvv[3]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==3) data.V22_rd[tt]+=corr_re[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv_rest[1],wsd_rnd_cvv[2]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==3) data.V22_rd[tt]-=corr_re[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv_rest[2],wsd_rnd_cvv[1]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==3) data.V22_rd[tt]+=corr_re[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv_rest[3],wsd_rnd_cvv[0]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==3) data.V22_rd[tt]-=corr_re[t]/norm;
            }
            /* V_3j */
            slices(j*10,wsd_rnd_cvv_rest[0],wsd_rnd_cvv[2]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==4) data.V33_rd[tt]-=corr_im[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv_rest[1],wsd_rnd_cvv[3]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==4) data.V33_rd[tt]+=corr_im[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv_rest[2],wsd_rnd_cvv[0]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==4) data.V33_rd[tt]-=corr_im[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv_rest[3],wsd_rnd_cvv[1]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==4) data.V33_rd[tt]+=corr_im[t]/norm;
            }
        }
        /*Calculate stochastic observables without distillation. Vector-Vector formed from the explicit spin-structure solutions.*/
        slices(0,wsd_rnd[1],wsd_rnd[ifl2]);
        for (t=0;t<file_head.tmax;++t)
        {
            tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
            data.P[tt]+=corr_re[t]/norm;
        }
        
        slices(1,wsd_rnd[1],wsd_rnd[ifl2]);
        for (t=0;t<file_head.tmax;++t)
        {
            tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
            data.A0[tt]+=corr_re[t]/norm;
        }
        
        slices(2,wsd_rnd[1],wsd_rnd[ifl2]);
        for (t=0;t<file_head.tmax;++t)
        {
            tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
            data.A1[tt]+=corr_im[t]/norm;
        }
        
        slices(3,wsd_rnd[1],wsd_rnd[ifl2]);
        for (t=0;t<file_head.tmax;++t)
        {
            tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
            data.A2[tt]+=corr_im[t]/norm;
        }
        
        slices(4,wsd_rnd[1],wsd_rnd[ifl2]);
        for (t=0;t<file_head.tmax;++t)
        {
            tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
            data.A3[tt]+=corr_im[t]/norm;
        }
        
        for(j=1;j<5;j++){
            /* V_0j */
            slices(j*10,wsd_rnd_cvv[0],wsd_rnd_cvv[2]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V00[tt]+=corr_re[t]/norm;
                if(j==2) data.V01[tt]+=corr_re[t]/norm;
                if(j==3) data.V02[tt]+=corr_re[t]/norm;
                if(j==4) data.V03[tt]+=corr_re[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv[1],wsd_rnd_cvv[3]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V00[tt]+=corr_re[t]/norm;
                if(j==2) data.V01[tt]+=corr_re[t]/norm;
                if(j==3) data.V02[tt]+=corr_re[t]/norm;
                if(j==4) data.V03[tt]+=corr_re[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv[2],wsd_rnd_cvv[0]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V00[tt]-=corr_re[t]/norm;
                if(j==2) data.V01[tt]-=corr_re[t]/norm;
                if(j==3) data.V02[tt]-=corr_re[t]/norm;
                if(j==4) data.V03[tt]-=corr_re[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv[3],wsd_rnd_cvv[1]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V00[tt]-=corr_re[t]/norm;
                if(j==2) data.V01[tt]-=corr_re[t]/norm;
                if(j==3) data.V02[tt]-=corr_re[t]/norm;
                if(j==4) data.V03[tt]-=corr_re[t]/norm;
            }
            
            /* V_1j */
            slices(j*10,wsd_rnd_cvv[0],wsd_rnd_cvv[3]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V10[tt]-=corr_im[t]/norm;
                if(j==2) data.V11[tt]-=corr_im[t]/norm;
                if(j==3) data.V12[tt]-=corr_im[t]/norm;
                if(j==4) data.V13[tt]-=corr_im[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv[1],wsd_rnd_cvv[2]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V10[tt]-=corr_im[t]/norm;
                if(j==2) data.V11[tt]-=corr_im[t]/norm;
                if(j==3) data.V12[tt]-=corr_im[t]/norm;
                if(j==4) data.V13[tt]-=corr_im[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv[2],wsd_rnd_cvv[1]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V10[tt]-=corr_im[t]/norm;
                if(j==2) data.V11[tt]-=corr_im[t]/norm;
                if(j==3) data.V12[tt]-=corr_im[t]/norm;
                if(j==4) data.V13[tt]-=corr_im[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv[3],wsd_rnd_cvv[0]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V10[tt]-=corr_im[t]/norm;
                if(j==2) data.V11[tt]-=corr_im[t]/norm;
                if(j==3) data.V12[tt]-=corr_im[t]/norm;
                if(j==4) data.V13[tt]-=corr_im[t]/norm;
            }
            
            /* V_2j */
            slices(j*10,wsd_rnd_cvv[0],wsd_rnd_cvv[3]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V20[tt]+=corr_re[t]/norm;
                if(j==2) data.V21[tt]+=corr_re[t]/norm;
                if(j==3) data.V22[tt]+=corr_re[t]/norm;
                if(j==4) data.V23[tt]+=corr_re[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv[1],wsd_rnd_cvv[2]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V20[tt]-=corr_re[t]/norm;
                if(j==2) data.V21[tt]-=corr_re[t]/norm;
                if(j==3) data.V22[tt]-=corr_re[t]/norm;
                if(j==4) data.V23[tt]-=corr_re[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv[2],wsd_rnd_cvv[1]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V20[tt]+=corr_re[t]/norm;
                if(j==2) data.V21[tt]+=corr_re[t]/norm;
                if(j==3) data.V22[tt]+=corr_re[t]/norm;
                if(j==4) data.V23[tt]+=corr_re[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv[3],wsd_rnd_cvv[0]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V20[tt]-=corr_re[t]/norm;
                if(j==2) data.V21[tt]-=corr_re[t]/norm;
                if(j==3) data.V22[tt]-=corr_re[t]/norm;
                if(j==4) data.V23[tt]-=corr_re[t]/norm;
            }
            /* V_3j */
            slices(j*10,wsd_rnd_cvv[0],wsd_rnd_cvv[2]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V30[tt]-=corr_im[t]/norm;
                if(j==2) data.V31[tt]-=corr_im[t]/norm;
                if(j==3) data.V32[tt]-=corr_im[t]/norm;
                if(j==4) data.V33[tt]-=corr_im[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv[1],wsd_rnd_cvv[3]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V30[tt]+=corr_im[t]/norm;
                if(j==2) data.V31[tt]+=corr_im[t]/norm;
                if(j==3) data.V32[tt]+=corr_im[t]/norm;
                if(j==4) data.V33[tt]+=corr_im[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv[2],wsd_rnd_cvv[0]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V30[tt]-=corr_im[t]/norm;
                if(j==2) data.V31[tt]-=corr_im[t]/norm;
                if(j==3) data.V32[tt]-=corr_im[t]/norm;
                if(j==4) data.V33[tt]-=corr_im[t]/norm;
            }
            slices(j*10,wsd_rnd_cvv[3],wsd_rnd_cvv[1]);
            for (t=0;t<file_head.tmax;++t)
            {
                tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
                if(j==1) data.V30[tt]+=corr_im[t]/norm;
                if(j==2) data.V31[tt]+=corr_im[t]/norm;
                if(j==3) data.V32[tt]+=corr_im[t]/norm;
                if(j==4) data.V33[tt]+=corr_im[t]/norm;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        wt2=MPI_Wtime();
        wt[3]+=(wt2-wt1);
        
        wtsum+=wt[0]+wt[1]+wt[2]+wt[3];
        if (my_rank==0){
            printf("Source n. %d, x0=%d, postprocessing time=%.2e\n",isrc,x0,wt[2]);
            printf("\n");
            printf("Source n. %d, x0=%d, observable contraction time=%.2e\n",isrc,x0,wt[3]);
            printf("\n");
            fflush(flog);
        }
        
    }
    /*Calculate total distilled observables as sum of the four contributions*/
    for (t=0;t<file_head.tmax;++t)
    {
        tt=(file_head.x0>=0)?t:((t-x0+file_head.tmax)%file_head.tmax);
        data.P_d[tt]+=data.P_dd[tt]+data.P_rr[tt]+data.P_rd[tt]+data.P_dr[tt];
        data.V00_d[tt]+=data.V00_dd[tt]+data.V00_rr[tt]+data.V00_rd[tt]+data.V00_dr[tt];
        data.V11_d[tt]+=data.V11_dd[tt]+data.V11_rr[tt]+data.V11_rd[tt]+data.V11_dr[tt];
        data.V22_d[tt]+=data.V22_dd[tt]+data.V22_rr[tt]+data.V22_rd[tt]+data.V22_dr[tt];
        data.V33_d[tt]+=data.V33_dd[tt]+data.V33_rr[tt]+data.V33_rd[tt]+data.V33_dr[tt];
        data.V03_d[tt]+=data.V03_dd[tt]+data.V03_rr[tt]+data.V03_rd[tt]+data.V03_dr[tt];
        
    }
    
    if(file_head.nfl==2)
        wtsum*=0.5;
    
    if (my_rank==0)
    {
        printf("Total time %.2e sec\n\n",wtsum);
        fflush(flog);
    }
    
    release_wsd();
    release_wsd();
    release_wsd();
    
    release_wsd();
    release_wsd();
    release_wsd();
    release_wsd();
    release_wsd();
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
        for (t=0;t<file_head.tmax;++t)
    printf("t = %3d  CP = %14.6e    CP_d = %14.6e    CP_dd = %14.6e CP_rr = %14.6e CP_dr = %14.6e CP_rd = %14.6e CP_t = %14.6e  CA_0 = %14.6e   CA_1 = %14.6e   CA_2 = %14.6e   CA_3 = %14.6e   V_00 = %14.6e   V_00_d = %14.6e   V_00_dd = %14.6e   V_00_rr = %14.6e   V_00_dr = %14.6e   V_00_rd = %14.6e  V_01 = %14.6e   V_02 = %14.6e   V_03 = %14.6e   V_03_d = %14.6e   V_03_dd = %14.6e   V_03_rr = %14.6e   V_03_dr = %14.6e   V_03_rd = %14.6e   V_10 = %14.6e   V_11 = %14.6e   V_11_d = %14.6e   V_11_dd = %14.6e   V_11_rr = %14.6e   V_11_dr = %14.6e   V_11_rd = %14.6e   V_12 = %14.6e   V_13 = %14.6e   V_20 = %14.6e   V_21 = %14.6e   V_22 = %14.6e   V_22_d = %14.6e   V_22_dd = %14.6e   V_22_rr = %14.6e   V_22_dr = %14.6e   V_22_rd = %14.6e   V_23 = %14.6e   V_30 = %14.6e   V_31 = %14.6e   V_32 = %14.6e   V_33 = %14.6e   V_33_d = %14.6e   V_33_dd = %14.6e   V_33_rr = %14.6e   V_33_dr = %14.6e   V_33_rd = %14.6e\n",
           t,data.P[t],data.P_d[t],data.P_dd[t],data.P_rr[t],data.P_dr[t],data.P_rd[t],data.P_t[t],data.A0[t],data.A1[t],data.A2[t],data.A3[t],data.V00[t],data.V00_d[t],data.V00_dd[t],data.V00_rr[t],data.V00_dr[t],data.V00_rd[t],data.V01[t],data.V02[t],data.V03[t],data.V03_d[t],data.V03_dd[t],data.V03_rr[t],data.V03_dr[t],data.V03_rd[t],data.V10[t],data.V11[t],data.V11_d[t],data.V11_dd[t],data.V11_rr[t],data.V11_dr[t],data.V11_rd[t],data.V12[t],data.V13[t],data.V20[t],data.V21[t],data.V22[t],data.V22_d[t],data.V22_dd[t],data.V22_rr[t],data.V22_dr[t],data.V22_rd[t],data.V23[t],data.V30[t],data.V31[t],data.V32[t],data.V33[t],data.V33_d[t],data.V33_dd[t],data.V33_rr[t],data.V33_dr[t],data.V33_rd[t]);
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

