#ifndef RSCF_TG_2D_CUDA_ADVANCE_H
#define RSCF_TG_2D_CUDA_ADVANCE_H
#include "globals.h"

__global__ void compute_star_velosity(Real *u, Real *v, Real *p, Real *u_star, Real *v_star, Real *fu, Real *fv, Real *gu, Real *gv);
__global__ void compute_p_rhs(Real *u_star, Real *v_star, Real *rhs_p);
__global__ void compute_delta_p(Real *p, Real *delta_p, Real *rhs_p);
__global__ void compute_p(Real *p, Real *delta_p);
__global__ void compute_velosity(Real *u, Real *v, Real *p);
void compute_single_step(Real *u, Real *v, Real *p, Real *fu, Real *fv, Real *gu, Real *gv, Real *u_star, Real *v_star, Real *rhs_p, Real *delta_p);
__global__ void compute_substep2_val(Real *u, Real *v, Real *p, Real *u_old, Real *v_old, Real *p_old);
__global__ void compute_substep3_val(Real *u, Real *v, Real *p, Real *u_old, Real *v_old, Real *p_old);

#endif //RSCF_TG_2D_CUDA_ADVANCE_H
