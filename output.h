#ifndef RSCF_TG_2D_CUDA_OUTPUT_H
#define RSCF_TG_2D_CUDA_OUTPUT_H
#include "globals.h"

void save_npz(int step);
void save_vtk(Real *u, Real *v, Real *p, int step);
void save(Real *u, Real *v, Real *p, int step);

#endif //RSCF_TG_2D_CUDA_OUTPUT_H
