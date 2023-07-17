#include "fluxes.h"
#include "reconstr.h"
#include "fluxes.h"


__global__
void compute_fluxes(Real *u, Real *v, Real *p, Real *fu, Real *fv, Real *gu, Real *gv) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i <= NX and j < NY) {
        int idx = j * (NX + 1) + i;
        int idxm = (K_WENO + j) * NXG + K_WENO + i - 1;
        int idxp = (K_WENO + j) * NXG + K_WENO + i;
        Real u_[2 * K_WENO], v_[2 * K_WENO];


        for (int p = -K_WENO + 1; p <= K_WENO; p++) {
            int pidx = (K_WENO + j) * NXG + K_WENO + i - 1 + p;
            u_[p + K_WENO - 1] = u[pidx];
            v_[p + K_WENO - 1] = v[pidx];
        }
        Real ul, vl, ur, vr;

        RECONSTR(u_, ul, ur);
        RECONSTR(v_, vl, vr);

        if (0.5*(u[idxm]+u[idxp]) > 0.) {
            fu[idx] = ul*ul;
            fv[idx] = ul*vl;
        }
        else {
            fu[idx] = ur*ur;
            fv[idx] = ur*vr;
        }
    }
    if (i < NX and j <= NY) {
        int idx = j * NX + i;
        int idxm = (K_WENO + j - 1) * NXG + K_WENO + i;
        int idxp = (K_WENO + j) * NXG + K_WENO + i;
        Real u_[2 * K_WENO], v_[2 * K_WENO];

        for (int p = -K_WENO + 1; p <= K_WENO; p++) {
            int pidx = (K_WENO + j - 1 + p) * NXG + K_WENO + i;
            u_[p + K_WENO - 1] = u[pidx];
            v_[p + K_WENO - 1] = v[pidx];
        }
        Real ul, vl, ur, vr;

        RECONSTR(u_, ul, ur);
        RECONSTR(v_, vl, vr);

        if (0.5*(v[idxm]+v[idxp]) > 0.) {
            gu[idx] = ul*vl;
            gv[idx] = vl*vl;
        }
        else {
            gu[idx] = ur*vr;
            gv[idx] = vr*vr;
        }
    }
}
