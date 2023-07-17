#ifndef RSCF_TG_2D_CUDA_RECONSTR_H
#define RSCF_TG_2D_CUDA_RECONSTR_H
#include "globals.h"
#include <cuda.h>

#define RECONSTR WENO5


__device__  Real sign(Real x);
__device__ Real minmod(Real x, Real y);
__device__ void CONST(Real *u, Real &ul, Real &ur);
__device__ void CONST(Real *u, Real &ul, Real &ur);
__device__ void TVD2(Real *u, Real &ul, Real &ur);
__device__ void WENO5(Real *u, Real &ul, Real &ur);

#endif //RSCF_TG_2D_CUDA_RECONSTR_H
