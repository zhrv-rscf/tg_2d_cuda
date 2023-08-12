%%file tg2d.cu
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


#define M_PI 3.14159265358979323846

#ifndef RECONSTR
#define RECONSTR WENO5
#endif

#define FLUX calc_flux_hllc

#define GAM 1.4

#ifndef K_WENO
#define K_WENO ( 3 )
#endif

#ifndef NX
#define NX 400
#endif

#define NY  NX
#define NXG  ( NX+2*K_WENO )
#define NYG  ( NY+2*K_WENO )
#define KDT  ((400)/(NX))


const int SAVE_STEP = 1000;
const int LOG_STEP = 1000;


#define BLOCK_SIZE 16


struct vec_t {
    Real x;
    Real y;
};

struct cons_t {
    Real ro;
    Real ru;
    Real rv;
    Real re;
};


__forceinline__ void _checkErr(cudaError cuda_err, int _line, std::string _file) {
    if (cuda_err != cudaSuccess) {
        printf("ERROR (file: %s, line: %d): %s \n",
               _file.c_str(), _line, cudaGetErrorString(cuda_err));
        abort();
    }
}

#define checkErr(f) _checkErr( f, __LINE__, __FILE__)


#define CUDA_CALL(SUBR, GR, TH, PARAM) do { SUBR<<<GR, TH>>>(PARAM); checkErr(cudaGetLastError()); } while(0)

template<typename T>
T *malloc_on_host(int nx) {
    return new T *[nx];
}

template<typename T>
T *malloc_on_device(int nx) {
    T *res_d;
    cudaMalloc(&res_d, sizeof(T) * nx);
    checkErr(cudaGetLastError());
    return res_d;
}


template<typename T>
T **malloc_on_host(int nx, int ny) {
    T **res_h;
    res_h = new T *[nx];
    for (int i = 0; i < nx; i++) {
        res_h[i] = new T[ny];
    }
    return res_h;
}

template<typename T>
T **malloc_on_device(int nx, int ny) {
    T **res_h, **res_d;
    res_h = new T *[nx];
    cudaMalloc(&res_d, sizeof(T *) * nx);
    checkErr(cudaGetLastError());
    for (int i = 0; i < nx; i++) {
        cudaMalloc(&(res_h[i]), sizeof(T) * ny);
        checkErr(cudaGetLastError());
    }
    cudaMemcpy(res_d, res_h, sizeof(T *) * nx, cudaMemcpyHostToDevice);
    checkErr(cudaGetLastError());
    delete[] res_h;
    return res_d;
}

template<typename T>
void copy_device_to_host(T **dest, T **src, int nx, int ny) {
    T **src_h = new T *[nx];
    cudaMemcpy(src_h, src, sizeof(T *) * nx, cudaMemcpyDeviceToHost);
    checkErr(cudaGetLastError());
    for (int i = 0; i < nx; i++) {
        cudaMemcpy(dest[i], src_h[i], sizeof(T) * ny, cudaMemcpyDeviceToHost);
        checkErr(cudaGetLastError());
    }
    delete[] src_h;
}

template<typename T>
void copy_host_to_device(T **dest, T **src, int nx, int ny) {
    T **dest_h = new T *[nx];
    cudaMemcpy(dest_h, dest, sizeof(T *) * nx, cudaMemcpyDeviceToHost);
    checkErr(cudaGetLastError());
    for (int i = 0; i < nx; i++) {
        cudaMemcpy(dest_h[i], src[i], sizeof(T) * ny, cudaMemcpyHostToDevice);
        checkErr(cudaGetLastError());
    }
    delete[] dest_h;
}

template<typename T>
void copy_device_to_device(T **dest, T **src, int nx, int ny) {
    T **src_h = new T *[nx];
    T **dest_h = new T *[nx];
    cudaMemcpy(src_h, src, sizeof(T *) * nx, cudaMemcpyDeviceToHost);
    checkErr(cudaGetLastError());
    cudaMemcpy(dest_h, dest, sizeof(T *) * nx, cudaMemcpyDeviceToHost);
    checkErr(cudaGetLastError());
    for (int i = 0; i < nx; i++) {
        cudaMemcpy(dest_h[i], src_h[i], sizeof(T) * ny, cudaMemcpyDeviceToDevice);
        checkErr(cudaGetLastError());
    }
    delete[] src_h;
    delete[] dest_h;
}

__device__
int get_i() {
    return blockDim.x * blockIdx.x + threadIdx.x;
}

__device__
int get_j() {
    return blockDim.y * blockIdx.y + threadIdx.y;
}


struct data_t {
    struct {
        cons_t **cons, **cons_old, **fluxx, **fluxy;
        vec_t **grad_u, **grad_v;
        vec_t **x;
    } d;
    struct {
        cons_t **cons;
        vec_t **x;
        Real **p_exact, **u_exact, **v_exact;
    } h;
    struct {
        Int step = 0;
        Real t = 0.;
        const Real L = 50./M_PI;
        const Real DLO_X = - M_PI*L;
        const Real DHI_X =   M_PI*L;
        const Real DLO_Y = - M_PI*L;
        const Real DHI_Y =   M_PI*L;


        const Real hx = (DHI_X - DLO_X) / NX;
        const Real hy = (DHI_Y - DLO_Y) / NY;
        const Real hx2 = 2. * hx;
        const Real hy2 = 2. * hy;
        const Real hx_pow2 = hx * hx;
        const Real hy_pow2 = hy * hy;

        const Real Ma = 0.1;
        const Real Re = 100.;
        const Real MU0 = 1.67e-5;
        const Real R = 297.;
        const Real T0 = 273.;
        const Real Cz = sqrt(GAM*R*T0);
        const Real U0 = Ma * Cz;
        const Real R0 = Re*MU0/(U0 * L);
        const Real P0 = R0*R*T0;

        const Real NU = MU0/R0;

        const Real CFL  = 0.5;
        const Real DT = 1.e-7 * KDT;
        const Real MAX_TIME = 0.003;//20.*L/U0;
    } g;

    data_t() {
        h.cons = malloc_on_host<cons_t>(NXG, NYG);
        h.x = malloc_on_host<vec_t>(NXG, NYG);

        h.p_exact = malloc_on_host<Real>(NXG, NYG);
        h.u_exact = malloc_on_host<Real>(NXG, NYG);
        h.v_exact = malloc_on_host<Real>(NXG, NYG);

        d.cons = malloc_on_device<cons_t>(NXG, NYG);
        d.cons_old = malloc_on_device<cons_t>(NXG, NYG);
        d.fluxx = malloc_on_device<cons_t>(NX + 1, NY);
        d.fluxy = malloc_on_device<cons_t>(NX, NY + 1);

        d.grad_u = malloc_on_device<vec_t>(NXG, NYG);
        d.grad_v = malloc_on_device<vec_t>(NXG, NYG);
        d.x = malloc_on_device<vec_t>(NXG, NYG);

    }

    void copy_to_host() {
        copy_device_to_host(h.cons, d.cons, NXG, NYG);
    }

    void copy_coord_to_host() {
        copy_device_to_host(h.x, d.x, NXG, NYG);
    }

    void copy_to_old() {
        copy_device_to_device(d.cons_old, d.cons, NXG, NYG);
    }

};


dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
dim3 grid(NX / threads.x + 1, NY / threads.y + 1);




Real prim_r[NX][NY];
Real prim_u[NX][NY];
Real prim_v[NX][NY];
Real prim_p[NX][NY];
Real prim_e[NX][NY];

Real exact_u[NX][NY];
Real exact_v[NX][NY];
Real exact_p[NX][NY];

__device__ __host__
Real _max_(Real a, Real b) {
return a > b ? a : b;
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
void init(data_t d) {
    int i = get_i();
    int j = get_j();
    if (i < NX and j < NY) {
        int ig = i + K_WENO;
        int jg = j + K_WENO;

        Real r, p, u, v;
        d.d.x[ig][jg].x = d.g.DLO_X + (i + Real(0.5)) * d.g.hx;
        d.d.x[ig][jg].y = d.g.DLO_Y + (j + Real(0.5)) * d.g.hy;


        u =   d.g.U0 * sin(d.d.x[ig][jg].x / d.g.L) * cos(d.d.x[ig][jg].y / d.g.L);
        v = - d.g.U0 * cos(d.d.x[ig][jg].x / d.g.L) * sin(d.d.x[ig][jg].y / d.g.L);
        p = d.g.P0 + (d.g.R0*d.g.U0*d.g.U0/8.) * (cos(2. * d.d.x[ig][jg].x / d.g.L) + cos(2. * d.d.x[ig][jg].y / d.g.L));
        r = p / (d.g.R * d.g.T0);


        d.d.cons[ig][jg].ro = r;
        d.d.cons[ig][jg].ru = r * u;
        d.d.cons[ig][jg].rv = r * v;
        d.d.cons[ig][jg].re = p / (GAM - 1.) + r * (u * u + v * v) * 0.5;
    }
}

template<typename T>
__global__
void fill_boundary(T **x) {
    int i = get_i();
    int j = get_j();
    if (i < K_WENO and j < NY) {
        x[K_WENO - i - 1][K_WENO + j] = x[NX + K_WENO - i - 1][K_WENO + j];
        x[NX + K_WENO + i][K_WENO + j] = x[K_WENO + i][K_WENO + j];
    }
    if (i < NX and j < K_WENO) {
        x[K_WENO + i][K_WENO - j - 1] = x[K_WENO + i][NY + K_WENO - j - 1];
        x[K_WENO + i][NY + K_WENO + j] = x[K_WENO + i][K_WENO + j];
    }
}



__device__
void TVD2(Real *u, Real &ul, Real &ur) {
    ul = u[K_WENO - 1] + 0.5 * minmod(u[K_WENO - 1] - u[K_WENO - 2], u[K_WENO] - u[K_WENO - 1]);
    ur = u[K_WENO] - 0.5 * minmod(u[K_WENO] - u[K_WENO - 1], u[K_WENO + 1] - u[K_WENO]);
}


__device__
void WENO5(Real *u, Real &ul, Real &ur) {
    Real beta[3];
    Real alpha[3];
    Real eps = 1.0e-6;
//    if ((u[2] - u[1]) * (u[3] - u[2]) < 0.0) ul = u[2];
//    else {
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
//    }
//    if ((u[3] - u[2]) * (u[4] - u[3]) < 0.0) ur = u[3];
//    else {
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
//    }
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


__device__
void calc_flux_lf(
        Real rl, Real pl, Real ul, Real vl, Real wl,
Real rr, Real pr, Real ur, Real vr, Real wr,
        Real &qr, Real &qu, Real &qv, Real &qw, Real &qe) {

Real cl = sqrt(GAM * pl / rl);
Real cr = sqrt(GAM * pr / rr);
Real alpha = _max_(fabs(ul) + cl, fabs(ur) + cr);

Real el = pl / (GAM - 1.) + 0.5 * rl * (ul * ul + vl * vl + wl * wl);
Real er = pr / (GAM - 1.) + 0.5 * rr * (ur * ur + vr * vr + wr * wr);

qr = 0.5 * (rl * ul + rr * ur - alpha * (rr - rl));
qu = 0.5 * (rl * ul * ul + pl + rr * ur * ur + pr - alpha * (rr * ur - rl * ul));
qv = 0.5 * (rl * ul * vl + rr * ur * vr - alpha * (rr * vr - rl * vl));
qw = 0.5 * (rl * ul * wl + rr * ur * wr - alpha * (rr * wr - rl * wl));
qe = 0.5 * ((el + pl) * ul + (er + pr) * ur - alpha * (er - el));
}


__global__
void compute_grad_fluxes(data_t d) {
    int i = get_i();
    int j = get_j();
    if (i <= NX and j < NY) {
        int ig = K_WENO + i;
        int jg = K_WENO + j;
        Real u_[2 * K_WENO], v_[2 * K_WENO];


        for (int p = -K_WENO + 1; p <= K_WENO; p++) {
            u_[p + K_WENO - 1] = d.d.cons[ig - 1 + p][jg].ru / d.d.cons[ig - 1 + p][jg].ro;
            v_[p + K_WENO - 1] = d.d.cons[ig - 1 + p][jg].rv / d.d.cons[ig - 1 + p][jg].ro;
        }
        Real ul, vl;
        Real ur, vr;

        RECONSTR(u_, ul, ur);
        RECONSTR(v_, vl, vr);

        d.d.fluxx[i][j].ru = 0.5 * (ul + ur);
        d.d.fluxx[i][j].rv = 0.5 * (vl + vr);

    }
    if (i < NX and j <= NY) {
        int ig = K_WENO + i;
        int jg = K_WENO + j;

        Real u_[2 * K_WENO], v_[2 * K_WENO];

        for (int p = -K_WENO + 1; p <= K_WENO; p++) {
            u_[p + K_WENO - 1] = d.d.cons[ig][jg - 1 + p].ru / d.d.cons[ig][jg - 1 + p].ro;
            v_[p + K_WENO - 1] = d.d.cons[ig][jg - 1 + p].rv / d.d.cons[ig][jg - 1 + p].ro;
        }
        Real ul, vl;
        Real ur, vr;

        RECONSTR(u_, ul, ur);
        RECONSTR(v_, vl, vr);

        d.d.fluxy[i][j].ru = 0.5 * (ul + ur);
        d.d.fluxy[i][j].rv = 0.5 * (vl + vr);
    }
}


__global__
void compute_grad(data_t d) {
    int i = get_i();
    int j = get_j();
    if (i < NX and j < NY) {
        d.d.grad_u[K_WENO + i][K_WENO + j].x = (d.d.fluxx[i + 1][j].ru - d.d.fluxx[i][j].ru) / d.g.hx;
        d.d.grad_v[K_WENO + i][K_WENO + j].x = (d.d.fluxx[i + 1][j].rv - d.d.fluxx[i][j].rv) / d.g.hx;

        d.d.grad_u[K_WENO + i][K_WENO + j].y = (d.d.fluxy[i][j + 1].ru - d.d.fluxy[i][j].ru) / d.g.hy;
        d.d.grad_v[K_WENO + i][K_WENO + j].y = (d.d.fluxy[i][j + 1].rv - d.d.fluxy[i][j].rv) / d.g.hy;
    }
}


__global__
void compute_fluxes(data_t d) {
    int i = get_i();
    int j = get_j();
    if (i <= NX and j < NY) {
        int ig = K_WENO + i;
        int jg = K_WENO + j;
        Real r_[2 * K_WENO], u_[2 * K_WENO], v_[2 * K_WENO], p_[2 * K_WENO];


        for (int p = -K_WENO + 1; p <= K_WENO; p++) {
            r_[p + K_WENO - 1] = d.d.cons[ig - 1 + p][jg].ro;
            u_[p + K_WENO - 1] = d.d.cons[ig - 1 + p][jg].ru / r_[p + K_WENO - 1];;
            v_[p + K_WENO - 1] = d.d.cons[ig - 1 + p][jg].rv / r_[p + K_WENO - 1];;
            p_[p + K_WENO - 1] = (GAM - 1.) * (d.d.cons[ig - 1 + p][jg].re -
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

        FLUX(rl, pl, ul, vl, 0,
             rr, pr, ur, vr, 0,
             qr, qu, qv, qw, qe);
        d.d.fluxx[i][j].ro = qr;
        d.d.fluxx[i][j].ru = qu;
        d.d.fluxx[i][j].rv = qv;
        d.d.fluxx[i][j].re = qe;

        // visc
        Real grad_u_x = 0.5 * (d.d.grad_u[ig - 1][jg].x + d.d.grad_u[ig][jg].x);
        Real grad_u_y = 0.5 * (d.d.grad_u[ig - 1][jg].y + d.d.grad_u[ig][jg].y);
        Real grad_v_x = 0.5 * (d.d.grad_v[ig - 1][jg].x + d.d.grad_v[ig][jg].x);
        Real grad_v_y = 0.5 * (d.d.grad_v[ig - 1][jg].y + d.d.grad_v[ig][jg].y);
        Real u = 0.5 * (u_[K_WENO - 1] + u_[K_WENO]);
        Real v = 0.5 * (v_[K_WENO - 1] + v_[K_WENO]);
        Real tau_xx = (- 2. * d.g.MU0 / 3.) * (grad_u_x + grad_v_y) + 2. * d.g.MU0 * grad_u_x;
        Real tau_yx = d.g.MU0 * (grad_u_y + grad_v_x);
        d.d.fluxx[i][j].ru -= tau_xx;
        d.d.fluxx[i][j].rv -= tau_yx;
        d.d.fluxx[i][j].re -= u * tau_xx + v * tau_yx;
    }
    if (i < NX and j <= NY) {
        int ig = K_WENO + i;
        int jg = K_WENO + j;

        Real r_[2 * K_WENO], u_[2 * K_WENO], v_[2 * K_WENO], p_[2 * K_WENO];

        for (int p = -K_WENO + 1; p <= K_WENO; p++) {
            r_[p + K_WENO - 1] = d.d.cons[ig][jg - 1 + p].ro;
            u_[p + K_WENO - 1] = d.d.cons[ig][jg - 1 + p].ru / r_[p + K_WENO - 1];;
            v_[p + K_WENO - 1] = d.d.cons[ig][jg - 1 + p].rv / r_[p + K_WENO - 1];;
            p_[p + K_WENO - 1] = (GAM - 1.) * (d.d.cons[ig][jg - 1 + p].re -
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

        FLUX(rl, pl, vl, 0, ul,
             rr, pr, vr, 0, ur,
             qr, qv, qw, qu, qe);
        d.d.fluxy[i][j].ro = qr;
        d.d.fluxy[i][j].ru = qu;
        d.d.fluxy[i][j].rv = qv;
        d.d.fluxy[i][j].re = qe;

        // visc
        Real grad_u_x = 0.5 * (d.d.grad_u[ig][jg - 1].x + d.d.grad_u[ig][jg].x);
        Real grad_u_y = 0.5 * (d.d.grad_u[ig][jg - 1].y + d.d.grad_u[ig][jg].y);
        Real grad_v_x = 0.5 * (d.d.grad_v[ig][jg - 1].x + d.d.grad_v[ig][jg].x);
        Real grad_v_y = 0.5 * (d.d.grad_v[ig][jg - 1].y + d.d.grad_v[ig][jg].y);
        Real u = 0.5 * (u_[K_WENO - 1] + u_[K_WENO]);
        Real v = 0.5 * (v_[K_WENO - 1] + v_[K_WENO]);
        Real tau_yy = ( - 2. * d.g.MU0 / 3.) * (grad_u_x + grad_v_y) + 2. * d.g.MU0 * grad_v_y;
        Real tau_xy = d.g.MU0 * (grad_u_y + grad_v_x);
        d.d.fluxy[i][j].ru -= tau_xy;
        d.d.fluxy[i][j].rv -= tau_yy;
        d.d.fluxy[i][j].re -= u * tau_xy + v * tau_yy;
    }
}


__global__
void compute_new_val(data_t d) {
    int i = get_i();
    int j = get_j();
    if (i < NX and j < NY) {

        // convective fluxes
        d.d.cons[K_WENO + i][K_WENO + j].ro -= d.g.DT * (
                (d.d.fluxx[i + 1][j].ro - d.d.fluxx[i][j].ro) / d.g.hx +
                (d.d.fluxy[i][j + 1].ro - d.d.fluxy[i][j].ro) / d.g.hy
        );
        d.d.cons[K_WENO + i][K_WENO + j].ru -= d.g.DT * (
                (d.d.fluxx[i + 1][j].ru - d.d.fluxx[i][j].ru) / d.g.hx +
                (d.d.fluxy[i][j + 1].ru - d.d.fluxy[i][j].ru) / d.g.hy
        );
        d.d.cons[K_WENO + i][K_WENO + j].rv -= d.g.DT * (
                (d.d.fluxx[i + 1][j].rv - d.d.fluxx[i][j].rv) / d.g.hx +
                (d.d.fluxy[i][j + 1].rv - d.d.fluxy[i][j].rv) / d.g.hy
        );
        d.d.cons[K_WENO + i][K_WENO + j].re -= d.g.DT * (
                (d.d.fluxx[i + 1][j].re - d.d.fluxx[i][j].re) / d.g.hx +
                (d.d.fluxy[i][j + 1].re - d.d.fluxy[i][j].re) / d.g.hy
        );

        // TODO
        //d.d.cons[K_WENO + i][K_WENO + j].ro = R0;

    }
}


void compute_single_step(data_t d) {
    CUDA_CALL(fill_boundary, grid, threads, d.d.cons);
    CUDA_CALL(compute_grad_fluxes, grid, threads, d);
    CUDA_CALL(compute_grad, grid, threads, d);
    CUDA_CALL(fill_boundary, grid, threads, d.d.grad_u);
    CUDA_CALL(fill_boundary, grid, threads, d.d.grad_v);
    CUDA_CALL(compute_fluxes, grid, threads, d);
    CUDA_CALL(compute_new_val, grid, threads, d);
}

__global__
void compute_substep2_val(data_t d) {
    int i = get_i();
    int j = get_j();
    if (i < NX and j < NY) {
        int ig = K_WENO + i;
        int jg = K_WENO + j;

        d.d.cons[ig][jg].ro *= 0.25;
        d.d.cons[ig][jg].ru *= 0.25;
        d.d.cons[ig][jg].rv *= 0.25;
        d.d.cons[ig][jg].re *= 0.25;

        d.d.cons[ig][jg].ro += 0.75 * d.d.cons_old[ig][jg].ro;
        d.d.cons[ig][jg].ru += 0.75 * d.d.cons_old[ig][jg].ru;
        d.d.cons[ig][jg].rv += 0.75 * d.d.cons_old[ig][jg].rv;
        d.d.cons[ig][jg].re += 0.75 * d.d.cons_old[ig][jg].re;
    }

}

__global__
void compute_substep3_val(data_t d) {
    int i = get_i();
    int j = get_j();
    if (i < NX and j < NY) {
        int ig = K_WENO + i;
        int jg = K_WENO + j;

        d.d.cons[ig][jg].ro *= 2.;
        d.d.cons[ig][jg].ru *= 2.;
        d.d.cons[ig][jg].rv *= 2.;
        d.d.cons[ig][jg].re *= 2.;

        d.d.cons[ig][jg].ro += d.d.cons_old[ig][jg].ro;
        d.d.cons[ig][jg].ru += d.d.cons_old[ig][jg].ru;
        d.d.cons[ig][jg].rv += d.d.cons_old[ig][jg].rv;
        d.d.cons[ig][jg].re += d.d.cons_old[ig][jg].re;

        d.d.cons[ig][jg].ro /= 3.;
        d.d.cons[ig][jg].ru /= 3.;
        d.d.cons[ig][jg].rv /= 3.;
        d.d.cons[ig][jg].re /= 3.;
    }

}


void compute_prim(data_t d) {
    d.copy_to_host();
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int ig = i + K_WENO;
            int jg = j + K_WENO;
            prim_r[i][j] = d.h.cons[ig][jg].ro;
            prim_u[i][j] = d.h.cons[ig][jg].ru / d.h.cons[ig][jg].ro;
            prim_v[i][j] = d.h.cons[ig][jg].rv / d.h.cons[ig][jg].ro;
            prim_e[i][j] = d.h.cons[ig][jg].re / prim_r[i][j] - 0.5 * (
                    prim_u[i][j] * prim_u[i][j] + prim_v[i][j] * prim_v[i][j]
            );
            prim_p[i][j] = (GAM - 1.) * prim_r[i][j] * prim_e[i][j];
        }
    }

}


void compute_exact(data_t d) {
    Real L  = d.g.L;
    Real L2 = d.g.L*d.g.L;
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int ig = i + K_WENO;
            int jg = j + K_WENO;
            exact_u[i][j] =   d.g.U0 * sin(d.h.x[ig][jg].x / L) * cos(d.h.x[ig][jg].y / L) * exp(-2. * d.g.NU * d.g.t / L2);
            exact_v[i][j] = - d.g.U0 * cos(d.h.x[ig][jg].x / L) * sin(d.h.x[ig][jg].y / L) * exp(-2. * d.g.NU * d.g.t / L2);
            exact_p[i][j] =   d.g.P0 + (d.g.R0*d.g.U0*d.g.U0/8.) *
                                       (cos(2. * d.h.x[ig][jg].x / L) + cos(2. * d.h.x[ig][jg].y / L)) * exp(-4. * d.g.NU * d.g.t / L2);
        }
    }

}


void save_npz(data_t d) {
    char fName[50];
    std::stringstream ss;
    ss << "res_" << std::setfill('0') << std::setw(5) << NX;

    strcpy(fName, ss.str().c_str());
    strcat(fName, ".npz");

    cnpy::npz_save(fName, "R", &(prim_r[0][0]), {NX, NY}, "w");
    cnpy::npz_save(fName, "U", &(prim_u[0][0]), {NX, NY}, "a");
    cnpy::npz_save(fName, "V", &(prim_v[0][0]), {NX, NY}, "a");
    cnpy::npz_save(fName, "P", &(prim_p[0][0]), {NX, NY}, "a");
    cnpy::npz_save(fName, "E", &(prim_e[0][0]), {NX, NY}, "a");

    cnpy::npz_save(fName, "U_exact", &(exact_u[0][0]), {NX, NY}, "a");
    cnpy::npz_save(fName, "V_exact", &(exact_v[0][0]), {NX, NY}, "a");
    cnpy::npz_save(fName, "P_exact", &(exact_p[0][0]), {NX, NY}, "a");
}


void save(data_t d) {
    compute_prim(d);
    compute_exact(d);
    save_npz(d);
}


void print_err(data_t d) {
    compute_prim(d);
    compute_exact(d);
    Real u_err_l1 = 0.;
    Real v_err_l1 = 0.;
    Real p_err_l1 = 0.;
    Real u_err_l2 = 0.;
    Real v_err_l2 = 0.;
    Real p_err_l2 = 0.;

    for (Int i = 0; i < NX; i++) {
        for (Int j = 0; j < NY; j++) {
            u_err_l1 += fabs(prim_u[i][j] - exact_u[i][j]);
            v_err_l1 += fabs(prim_v[i][j] - exact_v[i][j]);
            p_err_l1 += fabs(prim_p[i][j] - exact_p[i][j]);
            u_err_l2 += (prim_u[i][j] - exact_u[i][j]) * (prim_u[i][j] - exact_u[i][j]);
            v_err_l2 += (prim_v[i][j] - exact_v[i][j]) * (prim_v[i][j] - exact_v[i][j]);
            p_err_l2 += (prim_p[i][j] - exact_p[i][j]) * (prim_p[i][j] - exact_p[i][j]);
        }
    }
    u_err_l1 *= d.g.hx * d.g.hy;
    v_err_l1 *= d.g.hx * d.g.hy;
    p_err_l1 *= d.g.hx * d.g.hy;

    u_err_l2 *= d.g.hx * d.g.hy;
    v_err_l2 *= d.g.hx * d.g.hy;
    p_err_l2 *= d.g.hx * d.g.hy;

    u_err_l2 = sqrt(u_err_l2);
    v_err_l2 = sqrt(v_err_l2);
    p_err_l2 = sqrt(p_err_l2);

    std::cout << "N = " << NX << "\t t = " << d.g.t << std::endl;
    std::cout << "||u_err||_L1 = " << u_err_l1 << std::endl;
    std::cout << "||v_err||_L1 = " << v_err_l1 << std::endl;
    std::cout << "||p_err||_L1 = " << p_err_l1 << std::endl;
    std::cout << "||u_err||_L2 = " << u_err_l2 << std::endl;
    std::cout << "||v_err||_L2 = " << v_err_l2 << std::endl;
    std::cout << "||p_err||_L2 = " << p_err_l2 << std::endl;
}


int main() {
    data_t data;

    std::cout << "MAX_TIME = " << data.g.MAX_TIME << std::endl;
    std::cout << "NX = " << NX << "  NY = " << NY << std::endl;
    std::cout << "DT = " << data.g.DT << std::endl;

    CUDA_CALL(init, grid, threads, data);
    data.copy_coord_to_host();

    //save(data);

    data.g.t = 0.;
    data.g.step = 0;

    time_t begin, end;
    time(&begin);
    while (data.g.t < data.g.MAX_TIME) {
        data.g.t += data.g.DT;
        data.g.step++;
        data.copy_to_old();
        compute_single_step(data);
        compute_single_step(data);
        CUDA_CALL(compute_substep2_val, grid, threads, data);
        compute_single_step(data);
        CUDA_CALL(compute_substep3_val, grid, threads, data);

        cudaDeviceSynchronize();

        if (data.g.step % LOG_STEP == 0) {
            time(&end);
            time_t elapsed = end - begin;
            printf("%d: Time elapsed for %d steps: %ld seconds.\n", data.g.step, LOG_STEP, elapsed);
            time(&begin);
        }
        if (data.g.step % SAVE_STEP == 0) {
            //save(data);
            //print_err(data);
        }
    }

    save(data);
    print_err(data);

    return 0;
}
