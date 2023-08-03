//%%file tg2d.cu
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
#define FLUX calc_flux_lf

#define GAM 1.4
#define K_WENO  3
#define NX  200
#define NY  200
#define NXG (NX+2*K_WENO)
#define NYG (NY+2*K_WENO)

const int SAVE_STEP = 1000;
const int LOG_STEP = 1000;


#define BLOCK_SIZE 4

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
        Int step;
        Real t;
        Real hx, hy;
        Real hx2, hy2;
        Real hx_pow2, hy_pow2;
        Real T0;
        Real P0;
        Real R0;
        Real U0;
        Real Ma;
        Real MU0;
        Real NU;
        Real Re;
        Real L;
        Real DLO_X;
        Real DHI_X;
        Real DLO_Y;
        Real DHI_Y;
        Real CFL;
        Real DT;
        Real MAX_TIME;
//        Real GAM;
        Real R;
    } g;

    void init_g() {
        g.L = 0.005;

        g.DLO_X = -M_PI*g.L;
        g.DHI_X =  M_PI*g.L;
        g.DLO_Y = -M_PI*g.L;
        g.DHI_Y =  M_PI*g.L;
        g.CFL  = 0.5;
        g.DT = 0.0625e-6;
        g.MAX_TIME = 320.;
//        g.GAM  = 1.4;


        g.step = 0;
        g.t = 0.;
        g.hx = (g.DHI_X - g.DLO_X) / NX;
        g.hy = (g.DHI_Y - g.DLO_Y) / NY;
        g.hx2 = 2. * g.hx;
        g.hy2 = 2. * g.hy;
        g.hx_pow2 = g.hx * g.hx;
        g.hy_pow2 = g.hy * g.hy;

        g.Re = 100.;
        g.MU0 = 1.67e-5;
        g.R = 297.;
        g.T0 = 273.;
        g.Ma  = 0.1;
        g.U0 = g.Ma * 0.1;
        g.R0 = g.Re*g.MU0/(g.U0 * g.L);
        g.P0 = g.R0*g.R*g.T0;

        g.NU = g.MU0/g.R0;
    }

    data_t() {
        alloc();
        init_g();
    }

    void alloc() {
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
        int ii = i + K_WENO;
        int jj = j + K_WENO;

        Real r, p, u, v;
        d.d.x[ii][jj].x = d.g.DLO_X + (i + Real(0.5)) * d.g.hx;
        d.d.x[ii][jj].y = d.g.DLO_Y + (j + Real(0.5)) * d.g.hy;


        u = d.g.U0 * cos(d.d.x[ii][jj].x/d.g.L) * sin(d.d.x[ii][jj].y/d.g.L);
        v = - d.g.U0 * sin(d.d.x[ii][jj].x/d.g.L) * cos(d.d.x[ii][jj].y/d.g.L);
        p = d.g.P0 + (d.g.R0*d.g.U0*d.g.U0/16.) * (cos(2. * d.d.x[ii][jj].x/d.g.L) + cos(2. * d.d.x[ii][jj].y/d.g.L));
        r = p / (d.g.R * d.g.T0);


        d.d.cons[ii][jj].ro = r;
        d.d.cons[ii][jj].ru = r * u;
        d.d.cons[ii][jj].rv = r * v;
        d.d.cons[ii][jj].re = p / (GAM - 1.) + r * (u * u + v * v) * 0.5;
    }
}


__global__
void fill_boundary(cons_t **cons) {
    int i = get_i();
    int j = get_j();
    if (i < K_WENO and j < NY) {
        int jj = K_WENO + j;
        int i_src = NX + K_WENO - i - 1;
        int i_dst = K_WENO - i - 1;
        cons[i_dst][jj].ro = cons[i_src][jj].ro;
        cons[i_dst][jj].ru = cons[i_src][jj].ru;
        cons[i_dst][jj].rv = cons[i_src][jj].rv;
        cons[i_dst][jj].re = cons[i_src][jj].re;
        i_dst = NX + K_WENO + i;
        i_src = K_WENO + i;
        cons[i_dst][jj].ro = cons[i_src][jj].ro;
        cons[i_dst][jj].ru = cons[i_src][jj].ru;
        cons[i_dst][jj].rv = cons[i_src][jj].rv;
        cons[i_dst][jj].re = cons[i_src][jj].re;
    }
    if (i < NX and j < K_WENO) {
        int ii = K_WENO + i;
        int j_dst = (K_WENO - j - 1);
        int j_src = (NY + K_WENO - j - 1);
        cons[ii][j_dst].ro = cons[ii][j_src].ro;
        cons[ii][j_dst].ru = cons[ii][j_src].ru;
        cons[ii][j_dst].rv = cons[ii][j_src].rv;
        cons[ii][j_dst].re = cons[ii][j_src].re;
        j_dst = NY + K_WENO + j;
        j_src = K_WENO + j;
        cons[ii][j_dst].ro = cons[ii][j_src].ro;
        cons[ii][j_dst].ru = cons[ii][j_src].ru;
        cons[ii][j_dst].rv = cons[ii][j_src].rv;
        cons[ii][j_dst].re = cons[ii][j_src].re;
    }
}


__global__
void fill_boundary(vec_t **vec) {
    int i = get_i();
    int j = get_j();
    if (i < K_WENO and j < NY) {
        int jj = K_WENO + j;
        int i_src = NX + K_WENO - i - 1;
        int i_dst = K_WENO - i - 1;
        vec[i_dst][jj].x = vec[i_src][jj].x;
        vec[i_dst][jj].y = vec[i_src][jj].y;
        i_dst = NX + K_WENO + i;
        i_src = K_WENO + i;
        vec[i_dst][jj].x = vec[i_src][jj].x;
        vec[i_dst][jj].y = vec[i_src][jj].y;
    }
    if (i < NX and j < K_WENO) {
        int ii = K_WENO + i;
        int j_dst = (K_WENO - j - 1);
        int j_src = (NY + K_WENO - j - 1);
        vec[ii][j_dst].x = vec[ii][j_src].x;
        vec[ii][j_dst].y = vec[ii][j_src].y;
        j_dst = NY + K_WENO + j;
        j_src = K_WENO + j;
        vec[ii][j_dst].x = vec[ii][j_src].x;
        vec[ii][j_dst].y = vec[ii][j_src].y;
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
    int _i = get_i();
    int _j = get_j();
    if (_i <= NX and _j < NY) {
        int i = K_WENO + _i;
        int j = K_WENO + _j;
        Real u_[2 * K_WENO], v_[2 * K_WENO];


        for (int p = -K_WENO + 1; p <= K_WENO; p++) {
            u_[p + K_WENO - 1] = d.d.cons[i - 1 + p][j].ru / d.d.cons[i - 1 + p][j].ro;
            v_[p + K_WENO - 1] = d.d.cons[i - 1 + p][j].rv / d.d.cons[i - 1 + p][j].ro;
        }
        Real ul, vl;
        Real ur, vr;

        RECONSTR(u_, ul, ur);
        RECONSTR(v_, vl, vr);

        d.d.fluxx[_i][_j].ru = 0.5 * (ul + ur);
        d.d.fluxx[_i][_j].rv = 0.5 * (vl + vr);

    }
    if (_i < NX and _j <= NY) {
        int i = K_WENO + _i;
        int j = K_WENO + _j;

        Real u_[2 * K_WENO], v_[2 * K_WENO];

        for (int p = -K_WENO + 1; p <= K_WENO; p++) {
            u_[p + K_WENO - 1] = d.d.cons[i][j - 1 + p].ru / d.d.cons[i][j - 1 + p].ro;
            v_[p + K_WENO - 1] = d.d.cons[i][j - 1 + p].rv / d.d.cons[i][j - 1 + p].ro;
        }
        Real ul, vl;
        Real ur, vr;

        RECONSTR(u_, ul, ur);
        RECONSTR(v_, vl, vr);

        d.d.fluxy[_i][_j].ru = 0.5 * (ul + ur);
        d.d.fluxy[_i][_j].rv = 0.5 * (vl + vr);
    }
}


__global__
void compute_grad(data_t d) {
    int _i = get_i();
    int _j = get_j();
    if (_i < NX and _j < NY) {
        int i = K_WENO + _i;
        int j = K_WENO + _j;

        d.d.grad_u[i][j].x = (d.d.fluxx[_i + 1][j].ru - d.d.fluxx[_i][_j].ru) / d.g.hx;
        d.d.grad_v[i][j].x = (d.d.fluxx[_i + 1][j].rv - d.d.fluxx[_i][_j].rv) / d.g.hx;

        d.d.grad_u[i][j].y = (d.d.fluxy[_i][_j + 1].ru - d.d.fluxy[_i][_j].ru) / d.g.hy;
        d.d.grad_v[i][j].y = (d.d.fluxy[_i][_j + 1].rv - d.d.fluxy[_i][_j].rv) / d.g.hy;
    }
}


__global__
void compute_fluxes(data_t d) {
    int _i = get_i();
    int _j = get_j();
    if (_i <= NX and _j < NY) {
        int i = K_WENO + _i;
        int j = K_WENO + _j;
        Real r_[2 * K_WENO], u_[2 * K_WENO], v_[2 * K_WENO], p_[2 * K_WENO];


        for (int p = -K_WENO + 1; p <= K_WENO; p++) {
            r_[p + K_WENO - 1] = d.d.cons[i - 1 + p][j].ro;
            u_[p + K_WENO - 1] = d.d.cons[i - 1 + p][j].ru / r_[p + K_WENO - 1];;
            v_[p + K_WENO - 1] = d.d.cons[i - 1 + p][j].rv / r_[p + K_WENO - 1];;
            p_[p + K_WENO - 1] = (GAM - 1.) * (d.d.cons[i - 1 + p][j].re -
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
        d.d.fluxx[_i][_j].ro = qr;
        d.d.fluxx[_i][_j].ru = qu;
        d.d.fluxx[_i][_j].rv = qv;
        d.d.fluxx[_i][_j].re = qe;

        // visc
        Real grad_u_x = 0.5 * (d.d.grad_u[i - 1][j].x + d.d.grad_u[i][j].x);
        Real grad_u_y = 0.5 * (d.d.grad_u[i - 1][j].y + d.d.grad_u[i][j].y);
        Real grad_v_x = 0.5 * (d.d.grad_v[i - 1][j].x + d.d.grad_v[i][j].x);
        Real grad_v_y = 0.5 * (d.d.grad_v[i - 1][j].y + d.d.grad_v[i][j].y);
        Real u = 0.5 * (u_[K_WENO - 1] + u_[K_WENO]);
        Real v = 0.5 * (v_[K_WENO - 1] + v_[K_WENO]);
        Real tau_xx = (- 2. * d.g.MU0 / 3.) * (grad_u_x + grad_v_y) + 2. * d.g.MU0 * grad_u_x;
        Real tau_yx = d.g.MU0 * (grad_u_y + grad_v_x);
        d.d.fluxx[_i][_j].ru -= tau_xx;
        d.d.fluxx[_i][_j].rv -= tau_yx;
        d.d.fluxx[_i][_j].re -= u * tau_xx + v * tau_yx;
    }
    if (_i < NX and _j <= NY) {
        int i = K_WENO + _i;
        int j = K_WENO + _j;

//        int idx = j * NX + i;
//        int idxm = (K_WENO + j - 1) * NXG + K_WENO + i;
//        int idxp = (K_WENO + j) * NXG + K_WENO + i;
        Real r_[2 * K_WENO], u_[2 * K_WENO], v_[2 * K_WENO], p_[2 * K_WENO];

        for (int p = -K_WENO + 1; p <= K_WENO; p++) {
//            int pidx = (K_WENO + j - 1 + p) * NXG + K_WENO + i;
            r_[p + K_WENO - 1] = d.d.cons[i][j - 1 + p].ro;
            u_[p + K_WENO - 1] = d.d.cons[i][j - 1 + p].ru / r_[p + K_WENO - 1];;
            v_[p + K_WENO - 1] = d.d.cons[i][j - 1 + p].rv / r_[p + K_WENO - 1];;
            p_[p + K_WENO - 1] = (GAM - 1.) * (d.d.cons[i][j - 1 + p].re -
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
        d.d.fluxy[_i][_j].ro = qr;
        d.d.fluxy[_i][_j].ru = qu;
        d.d.fluxy[_i][_j].rv = qv;
        d.d.fluxy[_i][_j].re = qe;

        // visc
        Real grad_u_x = 0.5 * (d.d.grad_u[i][j - 1].x + d.d.grad_u[i][j].x);
        Real grad_u_y = 0.5 * (d.d.grad_u[i][j - 1].y + d.d.grad_u[i][j].y);
        Real grad_v_x = 0.5 * (d.d.grad_v[i][j - 1].x + d.d.grad_v[i][j].x);
        Real grad_v_y = 0.5 * (d.d.grad_v[i][j - 1].y + d.d.grad_v[i][j].y);
        Real u = 0.5 * (u_[K_WENO - 1] + u_[K_WENO]);
        Real v = 0.5 * (v_[K_WENO - 1] + v_[K_WENO]);
        Real tau_yy = ( - 2. * d.g.MU0 / 3.) * (grad_u_x + grad_v_y) + 2. * d.g.MU0 * grad_v_y;
        Real tau_xy = d.g.MU0 * (grad_u_y + grad_v_x);
        d.d.fluxy[_i][_j].ru -= tau_xy;
        d.d.fluxy[_i][_j].rv -= tau_yy;
        d.d.fluxy[_i][_j].re -= u * tau_xy + v * tau_yy;
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
    fill_boundary<<<grid, threads>>>(d.d.cons);
    checkErr(cudaGetLastError());
    compute_grad_fluxes<<<grid, threads>>>(d);
    checkErr(cudaGetLastError());
    compute_grad<<<grid, threads>>>(d);
    checkErr(cudaGetLastError());
    fill_boundary<<<grid, threads>>>(d.d.grad_u);
    checkErr(cudaGetLastError());
    fill_boundary<<<grid, threads>>>(d.d.grad_v);
    checkErr(cudaGetLastError());
    compute_fluxes<<<grid, threads>>>(d);
    checkErr(cudaGetLastError());
    compute_new_val<<<grid, threads>>>(d);
    checkErr(cudaGetLastError());
}

__global__
void compute_substep2_val(data_t d) {
    int _i = get_i();
    int _j = get_j();
    if (_i < NX and _j < NY) {
        int i = K_WENO + _i;
        int j = K_WENO + _j;

        d.d.cons[i][j].ro *= 0.25;
        d.d.cons[i][j].ru *= 0.25;
        d.d.cons[i][j].rv *= 0.25;
        d.d.cons[i][j].re *= 0.25;

        d.d.cons[i][j].ro += 0.75 * d.d.cons_old[i][j].ro;
        d.d.cons[i][j].ru += 0.75 * d.d.cons_old[i][j].ru;
        d.d.cons[i][j].rv += 0.75 * d.d.cons_old[i][j].rv;
        d.d.cons[i][j].re += 0.75 * d.d.cons_old[i][j].re;
    }

}

__global__
void compute_substep3_val(data_t d) {
    int _i = get_i();
    int _j = get_j();
    if (_i < NX and _j < NY) {
        int i = K_WENO + _i;
        int j = K_WENO + _j;

        d.d.cons[i][j].ro *= 2.;
        d.d.cons[i][j].ru *= 2.;
        d.d.cons[i][j].rv *= 2.;
        d.d.cons[i][j].re *= 2.;

        d.d.cons[i][j].ro += d.d.cons_old[i][j].ro;
        d.d.cons[i][j].ru += d.d.cons_old[i][j].ru;
        d.d.cons[i][j].rv += d.d.cons_old[i][j].rv;
        d.d.cons[i][j].re += d.d.cons_old[i][j].re;

        d.d.cons[i][j].ro /= 3.;
        d.d.cons[i][j].ru /= 3.;
        d.d.cons[i][j].rv /= 3.;
        d.d.cons[i][j].re /= 3.;
    }

}


void compute_prim(data_t d) {
    d.copy_to_host();
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int ii = i + K_WENO;
            int jj = j + K_WENO;
            prim_r[i][j] = d.h.cons[ii][jj].ro;
            prim_u[i][j] = d.h.cons[ii][jj].ru / d.h.cons[ii][jj].ro;
            prim_v[i][j] = d.h.cons[ii][jj].rv / d.h.cons[ii][jj].ro;
            prim_e[i][j] = d.h.cons[ii][jj].re / prim_r[i][j] - 0.5 * (
                    prim_u[i][j] * prim_u[i][j] + prim_v[i][j] * prim_v[i][j]
            );
            prim_p[i][j] = (GAM - 1.) * prim_r[i][j] * prim_e[i][j];
        }
    }

}


void compute_exact(data_t d) {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int ii = i + K_WENO;
            int jj = j + K_WENO;
            exact_u[i][j] = d.g.U0 * cos(d.h.x[ii][jj].x/d.g.L) * sin(d.h.x[ii][jj].y/d.g.L) * exp(-2. * d.g.NU * d.g.t);
            exact_v[i][j] = - d.g.U0 * sin(d.h.x[ii][jj].x/d.g.L) * cos(d.h.x[ii][jj].y/d.g.L) * exp(-2. * d.g.NU * d.g.t);
            exact_p[i][j] =
                    d.g.P0 + (d.g.R0*d.g.U0*d.g.U0/16.) * (cos(2. * d.h.x[ii][jj].x/d.g.L)
                    + cos(2. * d.h.x[ii][jj].y/d.g.L)) * exp(-4. * d.g.NU * d.g.t);
        }
    }

}


void save_npz(data_t d) {
    char fName[50];
    std::stringstream ss;
    ss << "res_" << std::setfill('0') << std::setw(10) << d.g.step;

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


int main() {
    data_t data;

    init<<<grid, threads>>>(data);
    checkErr(cudaGetLastError());
    data.copy_coord_to_host();

    save(data);

    data.g.t = 0.;
    data.g.step = 0;

    time_t begin, end;
    time(&begin);
    while (data.g.t < data.g.MAX_TIME) {
        data.g.t += data.g.DT;
        data.g.step++;
        data.copy_to_old();
        compute_single_step(data);
        checkErr(cudaGetLastError());
        compute_single_step(data);
        checkErr(cudaGetLastError());
        compute_substep2_val<<<grid, threads>>>(data);
        compute_single_step(data);
        checkErr(cudaGetLastError());
        compute_substep3_val<<<grid, threads>>>(data);

        cudaDeviceSynchronize();

        if (data.g.step % LOG_STEP == 0) {
            time(&end);
            time_t elapsed = end - begin;
            printf("%d: Time elapsed for %d steps: %ld seconds.\n", data.g.step, LOG_STEP, elapsed);
            time(&begin);
        }
        if (data.g.step % SAVE_STEP == 0) {
            save(data);
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
            u_err_l1 *= data.g.hx * data.g.hy;
            v_err_l1 *= data.g.hx * data.g.hy;
            p_err_l1 *= data.g.hx * data.g.hy;

            u_err_l2 *= data.g.hx * data.g.hy;
            v_err_l2 *= data.g.hx * data.g.hy;
            p_err_l2 *= data.g.hx * data.g.hy;

            u_err_l2 = sqrt(u_err_l2);
            v_err_l2 = sqrt(v_err_l2);
            p_err_l2 = sqrt(p_err_l2);

            std::cout << "N = " << NX << std::endl;
            std::cout << "||u_err||_L1 = " << u_err_l1 << std::endl;
            std::cout << "||v_err||_L1 = " << v_err_l1 << std::endl;
            std::cout << "||p_err||_L1 = " << p_err_l1 << std::endl;
            std::cout << "||u_err||_L2 = " << u_err_l2 << std::endl;
            std::cout << "||v_err||_L2 = " << v_err_l2 << std::endl;
            std::cout << "||p_err||_L2 = " << p_err_l2 << std::endl;
        }
    }

    save(data);


    return 0;
}
