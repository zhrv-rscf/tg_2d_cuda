#ifndef RSCF_TG_2D_CUDA_FLUXES_H
#define RSCF_TG_2D_CUDA_FLUXES_H
#include "globals.h"
#include <cuda.h>

__global__ void compute_fluxes(Real *u, Real *v, Real *p, Real *fu, Real *fv, Real *gu, Real *gv);

#endif //RSCF_TG_2D_CUDA_FLUXES_H
