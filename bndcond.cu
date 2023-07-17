#include "globals.h"

__global__
void fill_boundary(Real *fld) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int id_src, id_dst;
    if (i < K_WENO and j < NY) {
        fld[(K_WENO + j) * NXG + (K_WENO - i - 1)] = fld[(K_WENO + j) * NXG + (NX + K_WENO - i - 1)];
        fld[(K_WENO + j) * NXG + (NX + K_WENO + i)] = fld[(K_WENO + j) * NXG + (K_WENO + i)];
    }
    if (i < NX and j < K_WENO) {
        fld[(K_WENO - j - 1) * NXG + (K_WENO + i)] = fld[(NY + K_WENO - j - 1) * NXG + (K_WENO + i)];
        fld[(NY + K_WENO + j) * NXG + (K_WENO + i)] = fld[(K_WENO + j) * NXG + (K_WENO + i)];
    }
}

