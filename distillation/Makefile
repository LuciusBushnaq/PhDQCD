

all: rmxeq mkdep mkxeq
.PHONY: all


MAIN = ym1 qcd1 ms1 ms2 ms3 ms4 ms5 ms6 mxw1 iso1

ARCHIVE = archive sarchive

BLOCK = block blk_grid map_u2blk map_sw2blk map_s2blk

DFL = dfl_geometry dfl_multiple dfl_subspace ltl_gcr dfl_sap_gcr dfl_modes

DFT = dft4d dft_com dft_shuf fft small_dft

DIRAC = Dw_dble Dw Dw_bnd

FLAGS = flags action_parms dfl_parms dirac_parms force_parms hmc_parms \
        lat_parms mdint_parms rat_parms rw_parms sap_parms solver_parms \
		  dft4d_parms dft_parms

FORCES = force0 force1 force2 force3 force4 force5 force6 \
         frcfcts genfrc tmcg tmcgm xtensor

HFLDS = hflds

LATTICE = bcnds uidx ftidx geometry cstar

LINALG = salg salg_dble valg valg_dble liealg cmatrix_dble cmatrix dalg

LINSOLV = cgne mscg fgcr fgcr4vd

LITTLE = Aw_gen Aw_com Aw_ops Aw_dble Aw ltl_modes

MDFLDS = mdflds su3fcom u1fcom md_cstar u1mom_map u1mom_facc

RANDOM = ranlux ranlxs ranlxd gauss

RATFCTS = elliptic zolotarev ratfcts

SAP = sap_com sap_gcr sap blk_solv

SFLDS = sflds scom sdcom Pbnd Pbnd_dble

SU3FCTS = chexp su3prod su3ren cm3x3 random_su3

SW_TERM = pauli pauli_dble swflds sw_term

TCHARGE = ftcom ftensor tcharge ym_action

UFLDS = plaq_sum uflds udcom bstap ud_cstar

UPDATE = chrono mdsteps counters mdint hmc rwtm rwtmeo rwrat rwrtm

UTILS = endian error hsum ioutils mutils utils wspace

U1FLDS = adflds adcom u1flds ad_shift u1_plaq_sum u1fcts u1_bstap ad_bcnds\
         ad_cstar

U1FTENSOR = u1ftcom u1ftensor u1fluxes mxw_action

VFLDS = vflds vinit vcom vdcom

WFLOW = wflow wflow_u1

LIB = dat_utils ms3_utils ms5_utils wflow_parms

COVLS = covlasolv

MODULES = $(ARCHIVE) $(BLOCK) $(DFL) $(DIRAC) $(FLAGS) $(FORCES) $(HFLDS)\
          $(LATTICE) $(LINALG) $(LINSOLV) $(LITTLE) $(MDFLDS) $(RANDOM) \
          $(RATFCTS) $(SAP) $(SFLDS) $(SU3FCTS) $(SW_TERM) $(TCHARGE) \
          $(UFLDS) $(UPDATE) $(UTILS) $(U1FLDS) $(U1FTENSOR) $(VFLDS) $(WFLOW) \
			 $(DFT) $(LIB)



LOGOPTION =



MDIR = ../modules

PRIMME_DIR=/work/mjp/lucius_test/primme-master

VPATH = .:$(MDIR)/flags:$(MDIR)/lattice:$(MDIR)/archive:$(MDIR)/linalg:\
          $(MDIR)/random:$(MDIR)/uflds:$(MDIR)/mdflds:$(MDIR)/su3fcts:\
          $(MDIR)/utils:$(MDIR)/forces:$(MDIR)/sflds:$(MDIR)/dirac:\
	       $(MDIR)/sw_term:$(MDIR)/tcharge:$(MDIR)/block:$(MDIR)/sap:\
	       $(MDIR)/linsolv:$(MDIR)/dfl:$(MDIR)/vflds:$(MDIR)/little:\
          $(MDIR)/update:$(MDIR)/wflow:$(MDIR)/ratfcts:$(MDIR)/hflds:\
			 $(MDIR)/u1flds:$(MDIR)/u1ftensor:$(MDIR)/dft:lib:\
				$(MDIR)/COVLS:



INCPATH = $(MPI_INCLUDE) ../include $(PRIMME_DIR)/include



LIBS = m lapack blas primme

LIBPATH = $(MPI_HOME)/lib ${PRIMME_DIR}/lib



CFLAGS = -std=c89 -pedantic -fstrict-aliasing \
         -Wall -Wno-long-long -Wstrict-prototypes -Werror \
         -O #-mno-avx -Dx64 -DPM

COVLSCFLAGS=-O3
LFLAGS =


SHELL=/bin/bash
CC=mpicc
CLINKER=$(CC)

PGMS= $(MAIN) $(MODULES)








-include $(addsuffix .d,$(PGMS))
-include $(addsuffix .d,$(COVLS))



$(addsuffix .d,$(PGMS)): %.d: %.c Makefile
	@ $(GCC) -ansi $< -MM $(addprefix -I,$(INCPATH)) -o $@
$(addsuffix .d,$(COVLS)): %.d: %.c Makefile
	@ $(GCC) -ansi $< -MM $(addprefix -I,$(INCPATH)) -o $@

$(addsuffix .o,$(PGMS)): %.o: %.c Makefile
	$(CC) $< -c $(CFLAGS) $(LOGOPTION) $(addprefix -I,$(INCPATH))
$(addsuffix .o,$(COVLS)): %.o: %.c Makefile
	$(CC) $< -c $(COVLSCFLAGS) $(LOGOPTION) $(addprefix -I,$(INCPATH))


$(MAIN): %: %.o $(addsuffix .o,$(MODULES)) $(addsuffix .o,$(COVLS)) Makefile
	$(CLINKER) $< $(addsuffix .o,$(MODULES)) $(addsuffix .o,$(COVLS)) $(LFLAGS) $(LOGOPTION) \
        $(addprefix -L,$(LIBPATH)) $(addprefix -l,$(LIBS)) -o $@


mkxeq: $(MAIN)



rmxeq:
	@ -rm -f $(MAIN); \
        echo "delete old executables"


mkdep:  $(addsuffix .d,$(PGMS)) $(addsuffix .d,$(COVLS))
	@ echo "generate tables of dependencies"



clean:
	@ -rm -rf *.d *.o *.alog *.clog *.slog $(MAIN)
.PHONY: clean

