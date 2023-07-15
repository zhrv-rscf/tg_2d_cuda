#include "cnpy/cnpy.h"
#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include <cstring>
#include <cmath>
#include <sstream>
#include <iomanip>

typedef double Real;
typedef int Int;


#define M_E            2.7182818284590452354    /* e */
#define M_LOG2E        1.4426950408889634074    /* log_2 e */
#define M_LOG10E    0.43429448190325182765    /* log_10 e */
#define M_LN2        0.69314718055994530942    /* log_e 2 */
#define M_LN10        2.30258509299404568402    /* log_e 10 */
#define M_PI        3.14159265358979323846    /* pi */
#define M_PI_2        1.57079632679489661923    /* pi/2 */
#define M_PI_4        0.78539816339744830962    /* pi/4 */
#define M_1_PI        0.31830988618379067154    /* 1/pi */
#define M_2_PI        0.63661977236758134308    /* 2/pi */
#define M_2_SQRTPI    1.12837916709551257390    /* 2/sqrt(pi) */
#define M_SQRT2        1.41421356237309504880    /* sqrt(2) */
#define M_SQRT1_2    0.70710678118654752440    /* 1/sqrt(2) */

#define __max__(x, y) ((x) > (y) ? (x) : (y))

#define FLUX_HLLC

#define RECONSTR WENO5


#define R0 1.0
#define V0 1.0
#define MU 0.01
#define MU_L 0.0
#define L 1.0

#define GAM  1.4
#define K_WENO  3
#define NX  100
#define NY  100
#define NXG (NX+2*K_WENO)
#define NYG (NY+2*K_WENO)
#define DLO_X 0.0
#define DHI_X (2*M_PI)
#define DLO_Y 0.0
#define DHI_Y (2*M_PI)
#define CFL  0.5
#define DT 1.e-3
#define MAX_TIME 32.

const int SAVE_STEP = 1000;
const int LOG_STEP = 1000;


#define BLOCK_SIZE 4

dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
dim3 grid(NX / threads.x + 1, NY / threads.y + 1);


Real *u, *v, *p, *u_star, *v_star, *fu, fv, *gu, *gv;
Real *u_h, *v_h, *p_h;

__forceinline__ void _checkErr(cudaError cuda_err, int _line, std::string _file) {
    if (cuda_err != cudaSuccess) {
        printf("ERROR (file: %s, line: %d): %s \n",
               _file.c_str(), _line, cudaGetErrorString(cuda_err));
        abort();
    }
}

#define checkErr(f) _checkErr( f, __LINE__, __FILE__)


template<typename T>
__host__
Real *mallocFieldsOnDevice(int nx, int ny) {
    T *c;
    cudaError_t result;
    result = cudaMalloc(&c, sizeof(T) * nx * ny);
    checkErr(result);
    return c;
}


__device__ __forceinline__
Int getIdx(int i, int j) {
    return j * NXG + i;
}


__device__ __forceinline__
Int getIdxFlxX(int i, int j) {
    return j * (NXG + 1) + i;
}


__device__ __forceinline__
Int getIdxFlxY(int i, int j) {
    return j * NXG + i;
}


__device__
Real sign(Real x) {
    if (x < 0.) {
        return -1.;
    } else if (x > 0.) {
        return 1.;
    } else {
        return 0.;
    }
}


__device__
Real minmod(Real x, Real y) {
    if (sign(x) != sign(y)) return 0.;
    return sign(x) * (fabs(x) < fabs(y) ? fabs(x) : fabs(y));
}


__device__
void CONST(Real *u, Real &ul, Real &ur) {
    ul = u[K_WENO - 1];
    ur = u[K_WENO];
}


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


__device__
void TVD2(Real *u, Real &ul, Real &ur) {
    ul = u[K_WENO - 1] + 0.5 * minmod(u[K_WENO - 1] - u[K_WENO - 2], u[3] - u[K_WENO - 1]);
    ur = u[3] - 0.5 * minmod(u[K_WENO] - u[K_WENO - 1], u[K_WENO + 1] - u[K_WENO]);
}


__device__
void WENO5(Real *u, Real &ul, Real &ur) {
    Real beta[3];
    Real alpha[3];
    Real eps = 1.0e-6;
    if ((u[2] - u[1]) * (u[3] - u[2]) < 0.0) ul = u[2];
    else {
        //значение слева
        beta[0] = (13. / 12.) * (u[2] - 2 * u[3] + u[4]) * (u[2] - 2 * u[3] + u[4]) +
                  0.25 * (3 * u[2] - 4 * u[3] + u[4]) * (3 * u[2] - 4 * u[3] + u[4]);
        beta[1] = (13. / 12.) * (u[1] - 2 * u[2] + u[3]) * (u[1] - 2 * u[2] + u[3]) +
                  0.25 * (u[1] - u[3]) * (u[1] - u[3]);
        beta[2] = (13. / 12.) * (u[0] - 2 * u[1] + u[2]) * (u[0] - 2 * u[1] + u[2]) +
                  0.25 * (u[0] - 4 * u[1] + 3 * u[2]) * (u[0] - 4 * u[1] + 3 * u[2]);
        alpha[0] = 0.3 / ((eps + beta[0]) * (eps + beta[0]));
        alpha[1] = 0.6 / ((eps + beta[1]) * (eps + beta[1]));
        alpha[2] = 0.1 / ((eps + beta[2]) * (eps + beta[2]));
        ul = (alpha[0] * (2 * u[2] + 5 * u[3] - u[4]) + alpha[1] * (-u[1] + 5 * u[2] + 2 * u[3]) +
              alpha[2] * (2 * u[0] - 7 * u[1] + 11 * u[2])) / ((alpha[0] + alpha[1] + alpha[2]) * 6);
    }
    if ((u[3] - u[2]) * (u[4] - u[3]) < 0.0) ur = u[3];
    else {
        //значение справа
        beta[0] = (13. / 12.) * (u[3] - 2 * u[4] + u[5]) * (u[3] - 2 * u[4] + u[5]) +
                  0.25 * (3 * u[3] - 4 * u[4] + u[5]) * (3 * u[3] - 4 * u[4] + u[5]);
        beta[1] = (13. / 12.) * (u[2] - 2 * u[3] + u[4]) * (u[2] - 2 * u[3] + u[4]) +
                  0.25 * (u[2] - u[4]) * (u[2] - u[4]);
        beta[2] = (13. / 12.) * (u[1] - 2 * u[2] + u[3]) * (u[1] - 2 * u[2] + u[3]) +
                  0.25 * (u[1] - 4 * u[2] + 3 * u[3]) * (u[1] - 4 * u[2] + 3 * u[3]);
        alpha[0] = 0.1 / ((eps + beta[0]) * (eps + beta[0]));
        alpha[1] = 0.6 / ((eps + beta[1]) * (eps + beta[1]));
        alpha[2] = 0.3 / ((eps + beta[2]) * (eps + beta[2]));
        ur = (alpha[0] * (11 * u[3] - 7 * u[4] + 2 * u[5]) + alpha[1] * (2 * u[2] + 5 * u[3] - u[4]) +
              alpha[2] * (-u[1] + 5 * u[2] + 2 * u[3])) / ((alpha[0] + alpha[1] + alpha[2]) * 6);
    }
}


#define F_HLLC_U(UK, FK, SK, SS, PK, RK, VK) (((SS)*((SK)*(UK)-(FK)) + (SK)*( (PK)+(RK)*((SK)-(VK))*((SS)-(VK)) )) / ((SK)-(SS)))
#define F_HLLC_V(UK, FK, SK, SS, PK, RK, VK) (((SS)*((SK)*(UK)-(FK))) / ((SK)-(SS)))
#define F_HLLC_E(UK, FK, SK, SS, PK, RK, VK) (((SS)*((SK)*(UK)-(FK)) + (SK)*( (PK)+(RK)*((SK)-(VK))*((SS)-(VK)) )*(SS)) / ((SK)-(SS)))


__device__
void calc_flux_hllc(
        Real rl, Real pl, Real ul, Real vl, Real wl,
        Real rr, Real pr, Real ur, Real vr, Real wr,
        Real &qr, Real &qu, Real &qv, Real &qw, Real &qe) {
    Real sl, sr, p_star, s_star, p_pvrs, _ql, _qr, tmp, e_tot_l, e_tot_r, czl, czr;

    e_tot_l = pl / rl / (GAM - 1.) + 0.5 * (ul * ul + vl * vl + wl * wl);
    e_tot_r = pr / rr / (GAM - 1.) + 0.5 * (ur * ur + vr * vr + wr * wr);

    czl = sqrt(GAM * pl / rl);
    czr = sqrt(GAM * pr / rr);

    p_pvrs = 0.5 * (pl + pr) - 0.5 * (ur - ul) * 0.25 * (rl + rr) * (czl + czr);
    p_star = (p_pvrs > 0.) ? p_pvrs : 0.;

    _ql = (p_star <= pl) ? 1 : sqrt(1. + (GAM + 1.) * (p_star / pl - 1.) / (2. * GAM));
    _qr = (p_star <= pr) ? 1 : sqrt(1. + (GAM + 1.) * (p_star / pr - 1.) / (2. * GAM));

    sl = ul - czl * _ql;
    sr = ur + czr * _qr;

    if (sl > sr) {
        tmp = sl;
        sl = sr;
        sr = tmp;
    }

    s_star = pr - pl;
    s_star += rl * ul * (sl - ul);
    s_star -= rr * ur * (sr - ur);
    s_star /= (rl * (sl - ul) - rr * (sr - ur));

    if (s_star < sl) s_star = sl;
    if (s_star > sr) s_star = sr;


    if (!((sl <= s_star) && (s_star <= sr))) {
        //        amrex::Print() << "HLLC: inequality SL <= S* <= SR is FALSE." << "\n";
        //        abort();
        return;
    }

    if (sl >= 0.) {
        qr = rl * ul;
        qu = rl * ul * ul + pl;
        qv = rl * vl * ul;
        qw = rl * wl * ul;
        qe = (rl * e_tot_l + pl) * ul;
    } else if (sr <= 0.) {
        qr = rr * ur;
        qu = rr * ur * ur + pr;
        qv = rr * vr * ur;
        qw = rr * wr * ur;
        qe = (rr * e_tot_r + pr) * ur;
    } else {
        if (s_star >= 0) {
            qr = F_HLLC_V( /*  UK, FK, SK, SS, PK, RK, VK */
                    rl,
                    rl * ul,
                    sl, s_star, pl, rl, ul
            );
            qu = F_HLLC_U( /*  UK, FK, SK, SS, PK, RK, VK */
                    rl * ul,
                    rl * ul * ul + pl,
                    sl, s_star, pl, rl, ul
            );
            qv = F_HLLC_V( /*  UK, FK, SK, SS, PK, RK, VK */
                    rl * vl,
                    rl * ul * vl,
                    sl, s_star, pl, rl, ul
            );
            qw = F_HLLC_V( /*  UK, FK, SK, SS, PK, RK, VK */
                    rl * wl,
                    rl * ul * wl,
                    sl, s_star, pl, rl, ul
            );
            qe = F_HLLC_E( /*  UK, FK, SK, SS, PK, RK, VK */
                    rl * e_tot_l,
                    (rl * e_tot_l + pl) * ul,
                    sl, s_star, pl, rl, ul
            );
        } else {
            qr = F_HLLC_V( /*  UK, FK, SK, SS, PK, RK, VK */
                    rr,
                    rr * ur,
                    sr, s_star, pr, rr, ur
            );
            qu = F_HLLC_U( /*  UK, FK, SK, SS, PK, RK, VK */
                    rr * ur,
                    rr * ur * ur + pr,
                    sr, s_star, pr, rr, ur
            );
            qv = F_HLLC_V( /*  UK, FK, SK, SS, PK, RK, VK */
                    rr * vr,
                    rr * ur * vr,
                    sr, s_star, pr, rr, ur
            );
            qw = F_HLLC_V( /*  UK, FK, SK, SS, PK, RK, VK */
                    rr * wr,
                    rr * ur * wr,
                    sr, s_star, pr, rr, ur
            );
            qe = F_HLLC_E( /*  UK, FK, SK, SS, PK, RK, VK */
                    rr * e_tot_r,
                    (rr * e_tot_r + pr) * ur,
                    sr, s_star, pr, rr, ur
            );
        }
    }
}


__global__
void compute_fluxes(Real *u, Real *v, Real *p, Real *fu, Real *fv, Real *gu, Real *gv) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i <= NX and j < NY) {
        int idx = j * (NX + 1) + i;
        int idxm = (K_WENO + j) * NXG + K_WENO + i - 1;
        int idxp = (K_WENO + j) * NXG + K_WENO + i;
        Real r_[2 * K_WENO], u_[2 * K_WENO], v_[2 * K_WENO], p_[2 * K_WENO];


        for (int p = -K_WENO + 1; p <= K_WENO; p++) {
            int pidx = (K_WENO + j) * NXG + K_WENO + i - 1 + p;
            r_[p + K_WENO - 1] = cons[pidx].ro;
            u_[p + K_WENO - 1] = cons[pidx].ru / r_[p + K_WENO - 1];;
            v_[p + K_WENO - 1] = cons[pidx].rv / r_[p + K_WENO - 1];;
            p_[p + K_WENO - 1] = (GAM - 1.) * (cons[pidx].re -
                                               0.5 * r_[p + K_WENO - 1] * (u_[p + K_WENO - 1] * u_[p + K_WENO - 1]
                                                                           + v_[p + K_WENO - 1] * v_[p + K_WENO - 1]));
        }
        Real rl, pl, ul, vl;
        Real rr, pr, ur, vr;

        RECONSTR(r_, rl, rr);
        RECONSTR(p_, pl, pr);
        RECONSTR(u_, ul, ur);
        RECONSTR(v_, vl, vr);

        Real qr, qu, qv, qw, qe;

        calc_flux_hllc(
                rl, pl, ul, vl, 0,
                rr, pr, ur, vr, 0,
                qr, qu, qv, qw, qe);
        fluxx[idx].ro = qr;
        fluxx[idx].ru = qu;
        fluxx[idx].rv = qv;
        fluxx[idx].re = qe;

        // visc
        Real grad_u_x = 0.5 * (grad_u[idxm].x + grad_u[idxp].x);
        Real grad_u_y = 0.5 * (grad_u[idxm].y + grad_u[idxp].y);
        Real grad_v_x = 0.5 * (grad_v[idxm].x + grad_v[idxp].x);
        Real grad_v_y = 0.5 * (grad_v[idxm].y + grad_v[idxp].y);
        Real u = 0.5 * (u_[K_WENO - 1] + u_[K_WENO]);
        Real v = 0.5 * (v_[K_WENO - 1] + v_[K_WENO]);
        Real tau_xx = (MU_L - 2. * MU / 3.) * (grad_u_x + grad_v_y) + 2. * MU * grad_u_x;
        Real tau_yx = MU * (grad_u_y + grad_v_x);
        fluxx[idx].ru -= tau_xx;
        fluxx[idx].rv -= tau_yx;
        fluxx[idx].re -= u * tau_xx + v * tau_yx;
    }
    if (i < NX and j <= NY) {
        int idx = j * NX + i;
        int idxm = (K_WENO + j - 1) * NXG + K_WENO + i;
        int idxp = (K_WENO + j) * NXG + K_WENO + i;
        Real r_[2 * K_WENO], u_[2 * K_WENO], v_[2 * K_WENO], p_[2 * K_WENO];

        for (int p = -K_WENO + 1; p <= K_WENO; p++) {
            int pidx = (K_WENO + j - 1 + p) * NXG + K_WENO + i;
            r_[p + K_WENO - 1] = cons[pidx].ro;
            u_[p + K_WENO - 1] = cons[pidx].ru / r_[p + K_WENO - 1];;
            v_[p + K_WENO - 1] = cons[pidx].rv / r_[p + K_WENO - 1];;
            p_[p + K_WENO - 1] = (GAM - 1.) * (cons[pidx].re -
                                               0.5 * r_[p + K_WENO - 1] * (u_[p + K_WENO - 1] * u_[p + K_WENO - 1]
                                                                           + v_[p + K_WENO - 1] * v_[p + K_WENO - 1]));
        }
        Real rl, pl, ul, vl;
        Real rr, pr, ur, vr;

        RECONSTR(r_, rl, rr);
        RECONSTR(p_, pl, pr);
        RECONSTR(u_, ul, ur);
        RECONSTR(v_, vl, vr);

        Real qr, qw, qu, qv, qe;

        calc_flux_hllc(
                rl, pl, vl, 0, ul,
                rr, pr, vr, 0, ur,
                qr, qv, qw, qu, qe);
        fluxy[idx].ro = qr;
        fluxy[idx].ru = qu;
        fluxy[idx].rv = qv;
        fluxy[idx].re = qe;

        // visc
        Real grad_u_x = 0.5 * (grad_u[idxm].x + grad_u[idxp].x);
        Real grad_u_y = 0.5 * (grad_u[idxm].y + grad_u[idxp].y);
        Real grad_v_x = 0.5 * (grad_v[idxm].x + grad_v[idxp].x);
        Real grad_v_y = 0.5 * (grad_v[idxm].y + grad_v[idxp].y);
        Real u = 0.5 * (u_[K_WENO - 1] + u_[K_WENO]);
        Real v = 0.5 * (v_[K_WENO - 1] + v_[K_WENO]);
        Real tau_yy = (MU_L - 2. * MU / 3.) * (grad_u_x + grad_v_y) + 2. * MU * grad_v_y;
        Real tau_xy = MU * (grad_u_y + grad_v_x);
        fluxy[idx].ru -= tau_xy;
        fluxy[idx].rv -= tau_yy;
        fluxy[idx].re -= u * tau_xy + v * tau_yy;
    }
}


__global__
void compute_star_val(Real *u, Real *v, Real *p, Real *u_star, Real *v_star, Real *fu, Real *fv, Real *gu, Real *gv) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Real Re = L * V0 * R0 / MU;
        Real dx[] = {(DHI_X - DLO_X) / NX, (DHI_Y - DLO_Y) / NY};
        int id = (K_WENO + j) * NXG + K_WENO + i;
        int idxp = (K_WENO + j) * NXG + K_WENO + i;
        int idxm = (K_WENO + j) * NXG + K_WENO - i;
        int idyp = (K_WENO + j + 1) * NXG + K_WENO + i;
        int idym = (K_WENO + j - 1) * NXG + K_WENO + i;

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
void compute_p(Real *u, Real *v, Real *p, Real *u_star, Real *v_star, Real *fu, Real *fv, Real *gu, Real *gv) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Real Re = L * V0 * R0 / MU;
        Real dx[] = {(DHI_X - DLO_X) / NX, (DHI_Y - DLO_Y) / NY};
        int id = (K_WENO + j) * NXG + K_WENO + i;
        int idxp = (K_WENO + j) * NXG + K_WENO + i;
        int idxm = (K_WENO + j) * NXG + K_WENO - i;
        int idyp = (K_WENO + j + 1) * NXG + K_WENO + i;
        int idym = (K_WENO + j - 1) * NXG + K_WENO + i;


    }
}


void compute_single_step(cons_t *cons_d, cons_t *fluxx, cons_t *fluxy, vec_t *grad_u, vec_t *grad_v) {
    fill_boundary<<<grid, threads>>>(u); checkErr(cudaGetLastError());
    fill_boundary<<<grid, threads>>>(v); checkErr(cudaGetLastError());
    fill_boundary<<<grid, threads>>>(p); checkErr(cudaGetLastError());

    compute_grad<<<grid, threads>>>(grad_u, grad_v, cons_d);
    checkErr(cudaGetLastError());
    fill_boundary<<<grid, threads>>>(grad_u);
    checkErr(cudaGetLastError());
    fill_boundary<<<grid, threads>>>(grad_v);
    checkErr(cudaGetLastError());
    compute_fluxes<<<grid, threads>>>(fluxx, fluxy, cons_d, grad_u, grad_v);
    checkErr(cudaGetLastError());
    compute_new_val<<<grid, threads>>>(cons_d, fluxx, fluxy);
    checkErr(cudaGetLastError());

}

__global__
void compute_substep2_val(cons_t *cons, cons_t *cons_old) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        int id = (K_WENO + j) * NXG + K_WENO + i;

        cons[id].ro *= 0.25;
        cons[id].ru *= 0.25;
        cons[id].rv *= 0.25;
        cons[id].re *= 0.25;

        cons[id].ro += 0.75 * cons_old[id].ro;
        cons[id].ru += 0.75 * cons_old[id].ru;
        cons[id].rv += 0.75 * cons_old[id].rv;
        cons[id].re += 0.75 * cons_old[id].re;
    }

}

__global__
void compute_substep3_val(cons_t *cons, cons_t *cons_old) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        int id = (K_WENO + j) * NXG + K_WENO + i;

        cons[id].ro *= 2.;
        cons[id].ru *= 2.;
        cons[id].rv *= 2.;
        cons[id].re *= 2.;

        cons[id].ro += cons_old[id].ro;
        cons[id].ru += cons_old[id].ru;
        cons[id].rv += cons_old[id].rv;
        cons[id].re += cons_old[id].re;

        cons[id].ro /= 3.;
        cons[id].ru /= 3.;
        cons[id].rv /= 3.;
        cons[id].re /= 3.;
    }

}


void save_npz(cons_t *cons, int step) {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int idx = (K_WENO + j) * NXG + K_WENO + i;
            prim_r[i][j] = cons[idx].ro;
            prim_u[i][j] = cons[idx].ru / cons[idx].ro;
            prim_v[i][j] = cons[idx].rv / cons[idx].ro;
            prim_p[i][j] = (GAM - 1.) * (
                    cons[idx].re - 0.5 * prim_r[i][j] * (
                            prim_u[i][j] * prim_u[i][j] + prim_v[i][j] * prim_v[i][j]
                    )
            );
        }
    }
    char fName[50];
    std::stringstream ss;
    ss << "res_" << std::setfill('0') << std::setw(10) << step;

    strcpy(fName, ss.str().c_str());
    strcat(fName, ".npz");

    cnpy::npz_save(fName, "R", &(prim_r[0][0]), {NX, NY}, "w");
    cnpy::npz_save(fName, "U", &(prim_u[0][0]), {NX, NY}, "a");
    cnpy::npz_save(fName, "V", &(prim_v[0][0]), {NX, NY}, "a");
    cnpy::npz_save(fName, "P", &(prim_p[0][0]), {NX, NY}, "a");

}


void save_vtk(cons_t *cons, int step) {
    Real dx[] = {(DHI_X - DLO_X) / NX, (DHI_Y - DLO_Y) / NY};
    char fName[50];
    std::stringstream ss;
    ss << "res_" << std::setfill('0') << std::setw(10) << step;

    strcpy(fName, ss.str().c_str());
    strcat(fName, ".vtk");
    FILE *fp = fopen(fName, "w");
    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "results\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");

    int pCount = (NX + 1) * (NX + 1);
    fprintf(fp, "POINTS %d float\n", pCount);

    for (int j = 0; j <= NY; j++) {
        for (int i = 0; i <= NX; i++) {
            fprintf(fp, "%f %f %f  \n", DLO_X + dx[0] * i, DLO_Y + dx[1] * j, 0.);
        }
    }
    fprintf(fp, "\n");

    int cellsCount = NX * NY;

    fprintf(fp, "CELLS %d %d\n", cellsCount, 5 * cellsCount);
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            fprintf(fp, "4 %d %d %d %d  \n", j * (NX + 1) + i, j * (NX + 1) + i + 1, (j + 1) * (NX + 1) + i + 1,
                    (j + 1) * (NX + 1) + i);
        }
    }
    fprintf(fp, "\n");

    fprintf(fp, "CELL_TYPES %d\n", cellsCount);
    for (int i = 0; i < cellsCount; i++) fprintf(fp, "9\n");
    fprintf(fp, "\n");

    fprintf(fp, "CELL_DATA %d\n", cellsCount);

    fprintf(fp, "SCALARS Density float 1\nLOOKUP_TABLE default\n");
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int id = (K_WENO + j) * NXG + K_WENO + i;
            fprintf(fp, "%f ", cons[id].ro);
            fprintf(fp, "\n");
        }
    }

    fprintf(fp, "SCALARS Pressure float 1\nLOOKUP_TABLE default\n");
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int id = (K_WENO + j) * NXG + K_WENO + i;
            Real u = cons[id].ru / cons[id].ro;
            Real v = cons[id].rv / cons[id].ro;
            Real p = (cons[id].re - cons[id].ro * (u * u + v * v)) * (GAM - 1.);
            fprintf(fp, "%f ", p);
        }
        fprintf(fp, "\n");
    }

//    fprintf(fp, "SCALARS Pressure_exact float 1\nLOOKUP_TABLE default\n");
//    for (int i = lo[0]; i <= hi[0]; i++)
//    {
//        for (int j = lo[1]; j <= hi[1]; j++)
//        {
//            fprintf(fp, "%f ", p_exact[i][j]);
//        }
//        fprintf(fp, "\n");
//    }
//
//    fprintf(fp, "SCALARS Pressure_err float 1\nLOOKUP_TABLE default\n");
//    for (int i = lo[0]; i <= hi[0]; i++)
//    {
//        for (int j = lo[1]; j <= hi[1]; j++)
//        {
//            fprintf(fp, "%f ", fabs(p[i][j]-p_exact[i][j]));
//        }
//        fprintf(fp, "\n");
//    }

    fprintf(fp, "SCALARS Energy float 1\nLOOKUP_TABLE default\n");
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int id = (K_WENO + j) * NXG + K_WENO + i;
            fprintf(fp, "%f ", cons[id].re / cons[id].ro);
        }
        fprintf(fp, "\n");
    }

//    fprintf(fp, "SCALARS Mach_number float 1\nLOOKUP_TABLE default\n");
//    for (int i = lo[0]; i <= hi[0]; i++)
//    {
//        for (int j = lo[1]; j <= hi[1]; j++)
//        {
//            fprintf(fp, "%f ", sqrt((u[i][j]*u[i][j]+v[i][j]*v[i][j])/(GAM*p[i][j]/r[i][j]))  );
//        }
//        fprintf(fp, "\n");
//    }

    fprintf(fp, "VECTORS Velosity float\n");
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int id = (K_WENO + j) * NXG + K_WENO + i;
            fprintf(fp, "%f %f %f  ", cons[id].ru / cons[id].ro, cons[id].rv / cons[id].ro, 0.0);
        }
        fprintf(fp, "\n");
    }

//    fprintf(fp, "VECTORS Velosity_exact float\n");
//    for (int i = lo[0]; i <= hi[0]; i++)
//    {
//        for (int j = lo[1]; j <= hi[1]; j++)
//        {
//            fprintf(fp, "%f %f %f  ", u_exact[i][j], v_exact[i][j], 0.0);
//        }
//        fprintf(fp, "\n");
//    }

//    fprintf(fp, "VECTORS Velosity_err float\n");
//    for (int i = lo[0]; i <= hi[0]; i++)
//    {
//        for (int j = lo[1]; j <= hi[1]; j++)
//        {
//            fprintf(fp, "%f %f %f  ", fabs(u[i][j]-u_exact[i][j]), fabs(v[i][j]-v_exact[i][j]), 0.0);
//        }
//        fprintf(fp, "\n");
//    }

    fclose(fp);
    printf("File '%s' saved...\n", fName);


}


void save(cons_t *cons, int step) {
    save_vtk(cons, step);
    save_npz(cons, step);
}


int main() {
//    cudaError_t result;
    cons_h = new cons_t[NXG * NYG];
    cons_d = mallocFieldsOnDevice<Real>(NXG, NYG);
    cons_d_old = mallocFieldsOnDevice<Real>(NXG, NYG);
    fluxx = mallocFieldsOnDevice<Real>(NX + 1, NY);
    fluxy = mallocFieldsOnDevice<Real>(NX, NY + 1);

//    gradx = mallocFieldsOnDevice(NXG,   NYG);
//    grady = mallocFieldsOnDevice(NXG,   NYG);

    grad_u = mallocVectorsOnDevice(NXG, NYG);
    grad_v = mallocVectorsOnDevice(NXG, NYG);

    init<<<grid, threads>>>(cons_d);
    checkErr(cudaGetLastError());

    cudaMemcpy(cons_h, cons_d, sizeof(cons_t) * NXG * NYG, cudaMemcpyDeviceToHost);
    save(cons_h, 0);

    double t = 0.;
    int step = 0;
    time_t begin, end;
    time(&begin);
    while (t < MAX_TIME) {
        t += DT;
        ++step;
        cudaMemcpy(cons_d_old, cons_d, NXG * NYG * sizeof(cons_t), cudaMemcpyDeviceToDevice);
        compute_single_step(cons_d, fluxx, fluxy, grad_u, grad_v);
        checkErr(cudaGetLastError());
        compute_single_step(cons_d, fluxx, fluxy, grad_u, grad_v);
        checkErr(cudaGetLastError());
        compute_substep2_val<<<grid, threads>>>(cons_d, cons_d_old);
        compute_single_step(cons_d, fluxx, fluxy, grad_u, grad_v);
        checkErr(cudaGetLastError());
        compute_substep3_val<<<grid, threads>>>(cons_d, cons_d_old);

        cudaDeviceSynchronize();

        if (step % LOG_STEP == 0) {
            time(&end);
            time_t elapsed = end - begin;
            printf("%d: Time measured for %d steps: %ld seconds.\n", LOG_STEP, step, elapsed);
            time(&begin);
        }
        if (step % SAVE_STEP == 0) {
            cudaMemcpy(cons_h, cons_d, NXG * NYG * sizeof(cons_t), cudaMemcpyDeviceToHost);
            save(cons_h, step);
        }
    }

    cudaMemcpy(cons_h, cons_d, NXG * NYG * sizeof(cons_t), cudaMemcpyDeviceToHost);
    save(cons_h, step);

    return 0;
}
