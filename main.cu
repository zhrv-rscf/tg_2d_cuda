#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include <cstring>
#include <cmath>
#include <sstream>
#include <iomanip>
#include "cnpy/cnpy.h"

#define CUDA_DEBUG

typedef double Real;
typedef int Int;

#define EPS            1.0e-6

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


#define CUDA_CALL(SUBROUTINE) {SUBROUTINE; checkErr(cudaGetLastError());}

#define __max__(x, y) ((x) > (y) ? (x) : (y))

#define FLUX_HLLC


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

const int SAVE_STEP = 100;
const int LOG_STEP = 100;


#define BLOCK_SIZE 4

#define RECONSTR WENO5



#ifdef CUDA_DEBUG

__forceinline__ void _checkErr(cudaError cuda_err, int _line, std::string _file) {
    if (cuda_err != cudaSuccess) {
        printf("ERROR (file: %s, line: %d): %s \n",
               _file.c_str(), _line, cudaGetErrorString(cuda_err));
        abort();
    }
}
#define checkErr(f) _checkErr( f, __LINE__, __FILE__)

#else

#define checkErr(f)

#endif


/**
 *  Основные массивы данных
 */

static dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
static dim3 grid(NX / threads.x + 1, NY / threads.y + 1);
static Real *u, *v, *p, *u_star, *v_star, *fu, *fv, *gu, *gv, *rhs_p, *delta_p, *u_old, *v_old, *p_old;
static Real *u_h, *v_h, *p_h, *delta_p_h;

template<typename T>
__forceinline__ Real *mallocFieldsOnDevice(int nx, int ny) {
    T *c;
    cudaError_t result;
    result = cudaMalloc(&c, sizeof(T) * nx * ny);
    checkErr(result);
    return c;
}

void mem_alloc() {
    u_h = new Real[NXG * NYG];
    v_h = new Real[NXG * NYG];
    p_h = new Real[NXG * NYG];
    delta_p_h = new Real[NXG * NYG];
    u = mallocFieldsOnDevice<Real>(NXG, NYG);
    v = mallocFieldsOnDevice<Real>(NXG, NYG);
    p = mallocFieldsOnDevice<Real>(NXG, NYG);
    u_star = mallocFieldsOnDevice<Real>(NXG, NYG);
    v_star = mallocFieldsOnDevice<Real>(NXG, NYG);
    u_old = mallocFieldsOnDevice<Real>(NXG, NYG);
    v_old = mallocFieldsOnDevice<Real>(NXG, NYG);
    p_old = mallocFieldsOnDevice<Real>(NXG, NYG);
    fu = mallocFieldsOnDevice<Real>(NX + 1, NY);
    fv = mallocFieldsOnDevice<Real>(NX + 1, NY);
    gu = mallocFieldsOnDevice<Real>(NX, NY + 1);
    gv = mallocFieldsOnDevice<Real>(NX, NY + 1);

    rhs_p = mallocFieldsOnDevice<Real>(NXG, NYG);
    delta_p = mallocFieldsOnDevice<Real>(NXG, NYG);
}

template<typename T>
void copy_to_host(T *dest, T *src) {
    cudaMemcpy(dest, src, sizeof(T) * NXG * NYG, cudaMemcpyDeviceToHost);
}

void copy_fields_to_host() {
    copy_to_host(u_h, u);
    copy_to_host(v_h, v);
    copy_to_host(p_h, p);
}





//void copy_fields_to_host();
//
//template<typename T>
//void copy_to_host(T *dest, T *src);





//void mem_alloc();
//__global__ void compute_star_velosity(Real *u, Real *v, Real *p, Real *u_star, Real *v_star, Real *fu, Real *fv, Real *gu, Real *gv);
//__global__ void compute_p_rhs(Real *u_star, Real *v_star, Real *rhs_p);
//__global__ void compute_delta_p(Real *p, Real *delta_p, Real *rhs_p);
//__global__ void compute_p(Real *p, Real *delta_p);
//__global__ void compute_velosity(Real *u, Real *v, Real *p);
//void compute_single_step(Real *u, Real *v, Real *p, Real *fu, Real *fv, Real *gu, Real *gv, Real *u_star, Real *v_star, Real *rhs_p, Real *delta_p);
//__global__ void compute_substep2_val(Real *u, Real *v, Real *p, Real *u_old, Real *v_old, Real *p_old);
//__global__ void compute_substep3_val(Real *u, Real *v, Real *p, Real *u_old, Real *v_old, Real *p_old);
//
//void save(Real *u, Real *v, Real *p, int step);
//
//__global__ void init(Real *u, Real *v, Real *p);
//__global__ void fill_boundary(Real *fld);
//
//__device__  Real sign(Real x);
//__device__ Real minmod(Real x, Real y);
//__device__ void CONST(Real *u, Real &ul, Real &ur);
//__device__ void CONST(Real *u, Real &ul, Real &ur);
//__device__ void TVD2(Real *u, Real &ul, Real &ur);
//__device__ void WENO5(Real *u, Real &ul, Real &ur);


/**
 * Граничные условия и начальные данные
 */

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





/**
 * Вычисление потоков
 */

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


/**
 * Вычисление методом проекции
 */

__global__
void compute_star_velosity(Real *u, Real *v, Real *p, Real *u_star, Real *v_star, Real *fu, Real *fv, Real *gu, Real *gv) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Real Re = L * V0 * R0 / MU;
        Real dx[] = {(DHI_X - DLO_X) / NX, (DHI_Y - DLO_Y) / NY};
        int id = (K_WENO + j) * NXG + K_WENO + i;

        int idxp = (K_WENO + j) * NXG + K_WENO + i + 1;
        int idxm = (K_WENO + j) * NXG + K_WENO + i - 1;
        int idyp = (K_WENO + j + 1) * NXG + K_WENO + i;
        int idym = (K_WENO + j + 1) * NXG + K_WENO + i;

        int idfxp = j * (NX + 1) + i + 1;
        int idfxm = j * (NX + 1) + i;
        int idgyp = (j + 1) * NX + i;
        int idgym = j * NX + i;

        // convective fluxes
        u_star[id] = u[id] - DT * (
                (fu[idfxp] - fu[idfxm]) / dx[0] + (gu[idgyp] - gu[idgym]) / dx[1] -
                (
                        (u[idxp] - 2. * u[id] + u[idxm]) / (dx[0] * dx[0]) -
                        (u[idyp] - 2. * u[id] + u[idym]) / (dx[1] * dx[1])
                ) / Re
        );
        v_star[id] = v[id] - DT * (
                (fv[idfxp] - fv[idfxm]) / dx[0] + (gv[idgyp] - gv[idgym]) / dx[1] -
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
        int idxm = (K_WENO + j) * NXG + K_WENO + i - 1;
        int idxp = (K_WENO + j) * NXG + K_WENO + i;
        Real u_[2 * K_WENO];


        for (int m = -K_WENO + 1; m <= K_WENO; m++) {
            int pidx = (K_WENO + j) * NXG + K_WENO + i - 1 + m;
            u_[m + K_WENO - 1] = u_star[pidx];
        }
        Real ul, ur;

        RECONSTR(u_, ul, ur);
        Real f = 0.5 * (ul + ur) / dx[0] / DT;

        rhs_p[idxp] -= f;
        rhs_p[idxm] += f;
    }
    if (i < NX and j <= NY) {
        int idxm = (K_WENO + j - 1) * NXG + K_WENO + i;
        int idxp = (K_WENO + j) * NXG + K_WENO + i;
        Real v_[2 * K_WENO];

        for (int m = -K_WENO + 1; m <= K_WENO; m++) {
            int pidx = (K_WENO + j - 1 + m) * NXG + K_WENO + i;
            v_[m + K_WENO - 1] = v_star[pidx];
        }
        Real vl, vr;

        RECONSTR(v_, vl, vr);
        Real f = 0.5 * (vl + vr) / dx[1] / DT;

        rhs_p[idxp] -= f;
        rhs_p[idxm] += f;
    }
}


__global__
void compute_delta_p(Real *p, Real *delta_p, Real *rhs_p) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
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
        p[id] += delta_p[id];
    }
}


__global__
void compute_velosity(Real *u, Real *v, Real *u_star, Real *v_star, Real *p) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Real dx[] = {(DHI_X - DLO_X) / NX, (DHI_Y - DLO_Y) / NY};
        int id = (K_WENO + j) * NXG + K_WENO + i;
        int idxm = (K_WENO + j) * NXG + K_WENO + i - 1;
        int idxp = (K_WENO + j) * NXG + K_WENO + i + 1;
        int idym = (K_WENO + j - 1) * NXG + K_WENO + i;
        int idyp = (K_WENO + j + 1) * NXG + K_WENO + i;

        u[id] = u_star[id] - DT*(p[idxp] - p[idxm]) / dx[0] / 2.;
        v[id] = v_star[id] - DT*(p[idyp] - p[idym]) / dx[1] / 2.;
    }
}

double compute_p_error() {
    copy_to_host(delta_p_h, delta_p);
    double err = 0.;
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int id = (K_WENO + j) * NXG + K_WENO + i;
            err += delta_p_h[id]*delta_p_h[id];
        }
    }
    err = sqrt(err);
    return err;
}

void compute_single_step(Real *u, Real *v, Real *p, Real *fu, Real *fv, Real *gu, Real *gv, Real *u_star, Real *v_star, Real *rhs_p, Real *delta_p) {
    compute_fluxes<<<grid, threads>>>(u, v, p, fu, fv, gu, gv); checkErr(cudaGetLastError());
    compute_star_velosity<<<grid, threads>>>(u, v, p, u_star, v_star, fu, fv, gu, gv); checkErr(cudaGetLastError());
    fill_boundary<<<grid, threads>>>(u_star); checkErr(cudaGetLastError());
    fill_boundary<<<grid, threads>>>(v_star); checkErr(cudaGetLastError());
    compute_p_rhs<<<grid, threads>>>(u_star, v_star, rhs_p); checkErr(cudaGetLastError());
    for (int iter = 0; iter < 1000; iter++) {
        compute_delta_p<<<grid, threads>>>(p, delta_p, rhs_p); checkErr(cudaGetLastError());
        compute_p<<<grid, threads>>>(p, delta_p); checkErr(cudaGetLastError());
        fill_boundary<<<grid, threads>>>(p); checkErr(cudaGetLastError());
        if (iter % 10 == 0) {
            if (compute_p_error() < EPS) break;
        }
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


Real p_out[NX][NY];
Real u_out[NX][NY];
Real v_out[NX][NY];



void save_npz(Real *u_h, Real *v_h, Real *p_h, int step) {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int id = (K_WENO + j) * NXG + K_WENO + i;
            u_out[i][j] = u_h[id];
            v_out[i][j] = v_h[id];
            p_out[i][j] = p_h[id];
        }
    }
    char fName[50];
    std::stringstream ss;
    ss << "tg_2d_inc_" << std::setfill('0') << std::setw(10) << step;

    strcpy(fName, ss.str().c_str());
    strcat(fName, ".npz");

    cnpy::npz_save(fName, "U", &(u_out[0][0]), {NX, NY}, "a");
    cnpy::npz_save(fName, "V", &(v_out[0][0]), {NX, NY}, "a");
    cnpy::npz_save(fName, "P", &(p_out[0][0]), {NX, NY}, "a");

}


void save_vtk(Real *u, Real *v, Real *p, int step) {
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
            fprintf(fp, "%f ", R0);
            fprintf(fp, "\n");
        }
    }

    fprintf(fp, "SCALARS Pressure float 1\nLOOKUP_TABLE default\n");
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int id = (K_WENO + j) * NXG + K_WENO + i;
            fprintf(fp, "%f ", p[id]);
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

//    fprintf(fp, "SCALARS Energy float 1\nLOOKUP_TABLE default\n");
//    for (int i = 0; i < NX; i++) {
//        for (int j = 0; j < NY; j++) {
//            int id = (K_WENO + j) * NXG + K_WENO + i;
//            fprintf(fp, "%f ", cons[id].re / cons[id].ro);
//        }
//        fprintf(fp, "\n");
//    }

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
            fprintf(fp, "%f %f %f  ", u[id], v[id], 0.0);
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


void save(Real *u, Real *v, Real *p, int step) {
    save_vtk(u, v, p, step);
    save_npz(u, v, p, step);
}

int main() {
//    cudaError_t result;

    mem_alloc();

    init<<<grid, threads>>>(u, v, p); checkErr(cudaGetLastError());

    copy_fields_to_host();
    save(u_h, v_h, p_h, 0);

    double t = 0.;
    int step = 0;
    time_t begin, end;
    time(&begin);
    while (t < MAX_TIME) {
        t += DT;
        ++step;
        cudaMemcpy(u_old, u, NXG * NYG * sizeof(Real), cudaMemcpyDeviceToDevice);
        cudaMemcpy(v_old, v, NXG * NYG * sizeof(Real), cudaMemcpyDeviceToDevice);
        cudaMemcpy(p_old, p, NXG * NYG * sizeof(Real), cudaMemcpyDeviceToDevice);
        compute_single_step(u, v, p, fu, fv, gu, gv, u_star, v_star, rhs_p, delta_p);
        compute_single_step(u, v, p, fu, fv, gu, gv, u_star, v_star, rhs_p, delta_p);
        compute_substep2_val<<<grid, threads>>>(u, v, p, u_old, v_old, p_old);
        compute_single_step(u, v, p, fu, fv, gu, gv, u_star, v_star, rhs_p, delta_p);
        compute_substep3_val<<<grid, threads>>>(u, v, p, u_old, v_old, p_old);

        cudaDeviceSynchronize();

        if (step % LOG_STEP == 0) {
            time(&end);
            time_t elapsed = end - begin;
            printf("%d: Time measured for %d steps: %ld seconds.\n", LOG_STEP, step, elapsed);
            time(&begin);
        }
        if (step % SAVE_STEP == 0) {

            save(u_h, v_h, p_h, step);
        }
    }

    copy_fields_to_host();
    save(u_h, v_h, p_h, step);

    return 0;
}
