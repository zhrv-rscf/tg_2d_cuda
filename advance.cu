#include "globals.h"
#include "advance.h"
#include "reconstr.h"
#include "fluxes.h"
#include "bndcond.h"

__global__
void compute_star_velosity(Real *u, Real *v, Real *p, Real *u_star, Real *v_star, Real *fu, Real *fv, Real *gu, Real *gv) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Real Re = L * V0 * R0 / MU;
        Real dx[] = {(DHI_X - DLO_X) / NX, (DHI_Y - DLO_Y) / NY};
        int id = (K_WENO + j) * NXG + K_WENO + i;
        int idxp = j * (NX + 1) + i + 1;
        int idxm = j * (NX + 1) + i;
        int idyp = (j + 1) * NX + i;
        int idym = j * NX + i;

        // convective fluxes
        u_star[id] = u[id] - DT * (
                (fu[idxp] - fu[idxm]) / dx[0] + (gu[idyp] - gu[idym]) / dx[1] -
                (
                        (u[idxp] - 2. * u[id] + u[idxm]) / (dx[0] * dx[0]) -
                        (u[idyp] - 2. * u[id] + u[idym]) / (dx[1] * dx[1])
                ) / Re
        );
        v_star[id] = v[id] - DT * (
                (fv[idxp] - fv[idxm]) / dx[0] + (gv[idyp] - gv[idym]) / dx[1] -
                (
                        (v[idxp] - 2. * v[id] + v[idxm]) / (dx[0] * dx[0]) -
                        (v[idyp] - 2. * v[id] + v[idym]) / (dx[1] * dx[1])
                ) / Re
        );
    }
}


__global__
void compute_p_rhs(Real *u_star, Real *v_star, Real *rhs_p) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    Real dx[] = {(DHI_X - DLO_X) / NX, (DHI_Y - DLO_Y) / NY};
    if (i <= NX and j < NY) {
        int idx = j * (NX + 1) + i;
        int idxm = (K_WENO + j) * NXG + K_WENO + i - 1;
        int idxp = (K_WENO + j) * NXG + K_WENO + i;
        Real u_[2 * K_WENO];


        for (int p = -K_WENO + 1; p <= K_WENO; p++) {
            int pidx = (K_WENO + j) * NXG + K_WENO + i - 1 + p;
            u_[p + K_WENO - 1] = u_star[pidx];
        }
        Real ul, vl, ur, vr;

        RECONSTR(u_, ul, ur);
        Real fu = 0.5*(ul+ur)/dx[0]/DT;

        rhs_p[idxp] -= fu;
        rhs_p[idxm] += fu;
    }
    if (i < NX and j <= NY) {
        int idx = j * NX + i;
        int idxm = (K_WENO + j - 1) * NXG + K_WENO + i;
        int idxp = (K_WENO + j) * NXG + K_WENO + i;
        Real v_[2 * K_WENO];

        for (int p = -K_WENO + 1; p <= K_WENO; p++) {
            int pidx = (K_WENO + j - 1 + p) * NXG + K_WENO + i;
            v_[p + K_WENO - 1] = v_star[pidx];
        }
        Real vl, vr;

        RECONSTR(v_, vl, vr);
        Real fu = 0.5*(vl+vr)/dx[1]/DT;

        rhs_p[idxp] -= fu;
        rhs_p[idxm] += fu;
    }
}


__global__
void compute_delta_p(Real *p, Real *delta_p, Real *rhs_p) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Real Re = L * V0 * R0 / MU;
        Real dx[] = {(DHI_X - DLO_X) / NX, (DHI_Y - DLO_Y) / NY};
        int id = (K_WENO + j) * NXG + K_WENO + i;
        int idxp = (K_WENO + j) * NXG + K_WENO + i + 1;
        int idxm = (K_WENO + j) * NXG + K_WENO + i - 1;
        int idyp = (K_WENO + j + 1) * NXG + K_WENO + i;
        int idym = (K_WENO + j - 1) * NXG + K_WENO + i;

        delta_p[id] = rhs_p[id] -
                      (p[idxp] - 2.*p[id] + p[idxm]) / (dx[0]*dx[0]) -
                      (p[idyp] - 2.*p[id] + p[idym]) / (dx[1]*dx[1]);
        delta_p[id] *= DT;
    }
}


__global__
void compute_p(Real *p, Real *delta_p) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
//        Real Re = L * V0 * R0 / MU;
//        Real dx[] = {(DHI_X - DLO_X) / NX, (DHI_Y - DLO_Y) / NY};
        int id = (K_WENO + j) * NXG + K_WENO + i;
        p[id] -= delta_p[id];
    }
}


__global__
void compute_velosity(Real *u, Real *v, Real *u_star, Real *v_star, Real *p) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Real Re = L * V0 * R0 / MU;
        Real dx[] = {(DHI_X - DLO_X) / NX, (DHI_Y - DLO_Y) / NY};
        int id = (K_WENO + j) * NXG + K_WENO + i;
        int idxm = (K_WENO + j) * NXG + K_WENO + i - 1;
        int idxp = (K_WENO + j) * NXG + K_WENO + i + 1;
        int idym = (K_WENO + j - 1) * NXG + K_WENO + i;
        int idyp = (K_WENO + j + 1) * NXG + K_WENO + i;

        u[id] = u_star[id] - DT*(p[idxp] - p[idxm]) / dx[0];
        v[id] = v_star[id] - DT*(p[idyp] - p[idym]) / dx[1];
    }
}


void compute_single_step() {

    compute_fluxes<<<grid, threads>>>(u, v, p, fu, fv, gu, gv); checkErr(cudaGetLastError());
    compute_star_velosity<<<grid, threads>>>(u, v, p, u_star, v_star, fu, fv, gu, gv); checkErr(cudaGetLastError());
    fill_boundary<<<grid, threads>>>(u_star); checkErr(cudaGetLastError());
    fill_boundary<<<grid, threads>>>(v_star); checkErr(cudaGetLastError());
    compute_p_rhs<<<grid, threads>>>(u_star, v_star, rhs_p); checkErr(cudaGetLastError());
    for (int iter = 0; iter < 100; iter++) {
        compute_delta_p<<<grid, threads>>>(p, delta_p, rhs_p); checkErr(cudaGetLastError());
        compute_p<<<grid, threads>>>(p, delta_p); checkErr(cudaGetLastError());
        fill_boundary<<<grid, threads>>>(p); checkErr(cudaGetLastError());
    }
    compute_velosity<<<grid, threads>>>(u, v, u_star, v_star, p); checkErr(cudaGetLastError());
    fill_boundary<<<grid, threads>>>(u); checkErr(cudaGetLastError());
    fill_boundary<<<grid, threads>>>(v); checkErr(cudaGetLastError());
}

__global__
void compute_substep2_val(Real *u, Real *v, Real *p, Real *u_old, Real *v_old, Real *p_old) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        int id = (K_WENO + j) * NXG + K_WENO + i;

        u[id] *= 0.25;
        v[id] *= 0.25;
        p[id] *= 0.25;

        u[id] += 0.75 * u_old[id];
        v[id] += 0.75 * v_old[id];
        p[id] += 0.75 * p_old[id];
    }

}

__global__
void compute_substep3_val(Real *u, Real *v, Real *p, Real *u_old, Real *v_old, Real *p_old) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        int id = (K_WENO + j) * NXG + K_WENO + i;

        u[id] *= 2.;
        v[id] *= 2.;
        p[id] *= 2.;

        u[id] += u_old[id];
        v[id] += v_old[id];
        p[id] += p_old[id];

        u[id] /= 3.;
        v[id] /= 3.;
        p[id] /= 3.;
    }

}
