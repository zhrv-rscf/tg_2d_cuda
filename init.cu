#include "init.h"

__global__
void init(Real *u, Real *v, Real *p) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        int idx = (j + K_WENO) * NXG + i + K_WENO;
        Real dx[] = {(DHI_X - DLO_X) / NX, (DHI_Y - DLO_Y) / NY};
        Real x[] = {
                DLO_X + (i + Real(0.5)) * dx[0],
                DLO_Y + (j + Real(0.5)) * dx[1]
        };

        u[idx] = cos(x[0]) * sin(x[1]);
        v[idx] = -sin(x[0]) * cos(x[1]);
        p[idx] = -0.25 * R0 * (cos(2. * x[0]) + cos(2. * x[1]));

    }
}
