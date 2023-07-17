#ifndef RSCF_TG_2D_CUDA_INIT_CUH
#define RSCF_TG_2D_CUDA_INIT_CUH
#include "globals.h"

__global__ void init(Real *u, Real *v, Real *p);

#endif //RSCF_TG_2D_CUDA_INIT_CUH
