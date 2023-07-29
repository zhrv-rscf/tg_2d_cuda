//%%file tg_2d_proj.cu

#include <cuda.h>
#include <string>
#include <iomanip>
#include "cnpy/cnpy.h"

typedef double  Real;
typedef int     Int;

#define PI 3.14159265358979323846

#define N_POINTS 801
#define L (2.*PI)
#define NU 0.01
#define RHO 1.0
#define N_ITERATIONS 32000

#define N_PRESSURE_POISSON_ITERATIONS 50
#define STABILITY_SAFETY_FACTOR 0.5

#define R0 1.0
#define V0 1.0
#define MU 0.01
#define MU_L 0.0

#define GAM  1.4
#define K_WENO  3
#define NX  N_POINTS
#define NY  N_POINTS
#define NXG (NX+2)
#define NYG (NY+2)
#define DLO_X 0.0
#define DHI_X L
#define DLO_Y 0.0
#define DHI_Y L
#define CFL  0.5
#define DT 1.e-3
#define MAX_TIME 32.

const int SAVE_STEP = 1000;
const int LOG_STEP = 1000;


#define BLOCK_SIZE 4

dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
dim3 grid(NX / threads.x + 1, NY / threads.y + 1);

__forceinline__ void _checkErr(cudaError cuda_err, int _line, std::string _file) {
    if (cuda_err != cudaSuccess) {
        printf("ERROR (file: %s, line: %d): %s \n",
               _file.c_str(), _line, cudaGetErrorString(cuda_err));
        abort();
    }
}

#define checkErr(f) _checkErr( f, __LINE__, __FILE__)

template<typename T>
T* mallocOnDevice(int nx, int ny) {
    T *c;
    cudaError_t result;
    result = cudaMalloc(&c, sizeof(T) * nx * ny);
    checkErr(result);
    return c;
}


template<typename T>
inline void copy_dev_to_dev(T *dest, T *src) {
    cudaMemcpy(dest, src, sizeof(T) * NXG * NYG, cudaMemcpyDeviceToDevice);
}


template<typename T>
void copy_host_to_dev(T *dest, T *src) {
    cudaMemcpy(dest, src, sizeof(T) * NXG * NYG, cudaMemcpyHostToDevice);
}


template<typename T>
void copy_dev_to_host(T *dest, T *src) {
    cudaMemcpy(dest, src, sizeof(T) * NXG * NYG, cudaMemcpyDeviceToHost);
}


struct data_t {
    struct {
        Real *u_prev, *v_prev, *p_prev;
        Real *u_tent, *v_tent;
        Real *u_next, *v_next, *p_next;
        Real *x, *y;
        Real *d_u_prev_d_x;
        Real *d_u_prev_d_y;
        Real *d_v_prev_d_x;
        Real *d_v_prev_d_y;
        Real *laplace_u_prev;
        Real *laplace_v_prev;
        Real *d_u_tent_d_x;
        Real *d_v_tent_d_y;
        Real *rhs;
        Real *d_p_next_d_x;
        Real *d_p_next_d_y;
    } d;

    struct {
        Real *u, *v, *p;
        Real *x, *y;
        Real *u_exact, *v_exact, *p_exact;
    } h;

    struct {
        Real hx, hy;
        Real t;
        Int step;
    } g;

    void mem_alloc() {
        h.u = new Real[NXG*NYG];
        h.v = new Real[NXG*NYG];
        h.p = new Real[NXG*NYG];

        h.u_exact = new Real[NXG*NYG];
        h.v_exact = new Real[NXG*NYG];
        h.p_exact = new Real[NXG*NYG];

        h.x = new Real[NXG*NYG];
        h.y = new Real[NXG*NYG];

        d.u_prev = mallocOnDevice<Real>(NXG, NYG);
        d.v_prev = mallocOnDevice<Real>(NXG, NYG);
        d.p_prev = mallocOnDevice<Real>(NXG, NYG);

        d.u_tent = mallocOnDevice<Real>(NXG, NYG);
        d.v_tent = mallocOnDevice<Real>(NXG, NYG);

        d.u_next = mallocOnDevice<Real>(NXG, NYG);
        d.v_next = mallocOnDevice<Real>(NXG, NYG);
        d.p_next = mallocOnDevice<Real>(NXG, NYG);

        d.x = mallocOnDevice<Real>(NXG, NYG);
        d.y = mallocOnDevice<Real>(NXG, NYG);

        d.d_u_prev_d_x = mallocOnDevice<Real>(NXG, NYG);
        d.d_u_prev_d_y = mallocOnDevice<Real>(NXG, NYG);
        d.d_v_prev_d_x = mallocOnDevice<Real>(NXG, NYG);
        d.d_v_prev_d_y = mallocOnDevice<Real>(NXG, NYG);

        d.laplace_u_prev = mallocOnDevice<Real>(NXG, NYG);
        d.laplace_v_prev = mallocOnDevice<Real>(NXG, NYG);

        d.d_u_tent_d_x = mallocOnDevice<Real>(NXG, NYG);
        d.d_v_tent_d_y = mallocOnDevice<Real>(NXG, NYG);

        d.rhs = mallocOnDevice<Real>(NXG, NYG);

        d.d_p_next_d_x = mallocOnDevice<Real>(NXG, NYG);
        d.d_p_next_d_y = mallocOnDevice<Real>(NXG, NYG);
    }

    void mem_free() {
        delete[] h.u;
        delete[] h.v;
        delete[] h.p;
    }

    void copy_to_host() {
        copy_dev_to_host<Real>(h.u, d.u_prev);
        copy_dev_to_host<Real>(h.v, d.v_prev);
        copy_dev_to_host<Real>(h.p, d.p_prev);
        copy_dev_to_host<Real>(h.x, d.x);
        copy_dev_to_host<Real>(h.y, d.y);
    }

    void calc_exact() {
        for (Int i = 0; i < NX; i++) {
            for (Int j = 0; j < NY; j++) {
                Int id = (j+1)*NXG+i+1;
                h.u_exact[id] =  cos(h.x[id]) * sin(h.y[id]) * exp(-2. * NU * g.t);
                h.v_exact[id] = -sin(h.x[id]) * cos(h.y[id]) * exp(-2. * NU * g.t);
                h.p_exact[id] = -0.25 * (cos(2. * h.x[id]) + cos(2. * h.y[id])) * exp(-4. * NU * g.t);
            }
        }
    }
};


__global__
void fill_bnd(Real *fld) {
    Int i = blockDim.x * blockIdx.x + threadIdx.x;
    Int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (j < NY) {
        fld[(j+1) * NXG + (0)] = fld[(1 + j) * NXG + (NX)];
        fld[(1 + j) * NXG + (NX + 1)] = fld[(1 + j) * NXG + (1)];
    }
    if (i < NX) {
        fld[1 + i] = fld[(NY) * NXG + (1 + i)];
        fld[(NY + 1) * NXG + (1 + i)] = fld[NXG + (1 + i)];
    }
}

__global__
void init(data_t d) {
    Int i = blockDim.x * blockIdx.x + threadIdx.x;
    Int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Int id = (j+1)*NXG+i+1;
        d.d.x[id] = DLO_X + i * d.g.hx;
        d.d.y[id] = DLO_Y + j * d.g.hy;

        d.d.u_prev[id] =   cos(d.d.x[id]) * sin(d.d.y[id]);
        d.d.v_prev[id] = - sin(d.d.x[id]) * cos(d.d.y[id]);
        d.d.p_prev[id] = - 0.25 * RHO * (cos(2. * d.d.x[id]) + cos(2. * d.d.y[id]));
    }
}

__global__
void central_difference_x(data_t d, Real *result, Real *src) {
    Int i = blockDim.x * blockIdx.x + threadIdx.x;
    Int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Int idm = (j+1)*NXG+i;
        Int id = (j+1)*NXG+i+1;
        Int idp = (j+1)*NXG+i+2;
        result[id] = (src[idp] - src[idm]) / (2*d.g.hx);
    }
}


__global__
void central_difference_y(data_t d, Real *result, Real *src) {
    Int i = blockDim.x * blockIdx.x + threadIdx.x;
    Int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Int idm = (j)*NXG+i+1;
        Int id = (j+1)*NXG+i+1;
        Int idp = (j+2)*NXG+i+1;
        result[id] = (src[idp] - src[idm]) / (2*d.g.hy);
    }
}


__global__
void laplace(data_t d, Real *result, Real *src) {
    Int i = blockDim.x * blockIdx.x + threadIdx.x;
    Int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Int id = (j + 1) * NXG + i + 1;
        Int idxm = (j + 1) * NXG + i;
        Int idxp = (j + 1) * NXG + i + 2;
        Int idym = (j - 0) * NXG + i + 1;
        Int idyp = (j + 2) * NXG + i + 1;
        result[id] = (src[idxp] - 2. * src[id] + src[idxm]) / (d.g.hx * d.g.hx) +
                     (src[idyp] - 2. * src[id] + src[idym]) / (d.g.hy * d.g.hy);
    }
}


__global__
void calc_tent_vel(data_t d) {
    Int i = blockDim.x * blockIdx.x + threadIdx.x;
    Int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Int id = (j + 1) * NXG + i + 1;
        d.d.u_tent[id] = d.d.u_prev[id] + DT * (NU * d.d.laplace_u_prev[id] - (d.d.u_prev[id] * d.d.d_u_prev_d_x[id] +
                                                                               d.d.v_prev[id] * d.d.d_u_prev_d_y[id]));
        d.d.v_tent[id] = d.d.v_prev[id] + DT * (NU * d.d.laplace_v_prev[id] - (d.d.u_prev[id] * d.d.d_v_prev_d_x[id] +
                                                                               d.d.v_prev[id] * d.d.d_v_prev_d_y[id]));
    }

}

__global__
void calc_p_rhs(data_t d) {
    Int i = blockDim.x * blockIdx.x + threadIdx.x;
    Int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Int id = (j + 1) * NXG + i + 1;
        d.d.rhs[id] =
                RHO / DT
                *
                (
                        d.d.d_u_tent_d_x[id]
                        +
                        d.d.d_v_tent_d_y[id]
                );
    }
}

__global__
void calc_p_next(data_t d) {
    Int i = blockDim.x * blockIdx.x + threadIdx.x;
    Int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Int id = (j + 1) * NXG + i + 1;
        Int idxm = (j + 1) * NXG + i - 0;
        Int idxp = (j + 1) * NXG + i + 2;
        Int idym = (j - 0) * NXG + i + 1;
        Int idyp = (j + 2) * NXG + i + 1;
        d.d.p_next[id] = 1. / 4. * (d.d.p_prev[idxm] + d.d.p_prev[idym] + d.d.p_prev[idxp] + d.d.p_prev[idyp]
                                    - d.g.hx * d.g.hx * d.d.rhs[id]);
    }
}


__global__
void calc_next_vel(data_t d) {
    Int i = blockDim.x * blockIdx.x + threadIdx.x;
    Int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NX and j < NY) {
        Int id = (j+1)*NXG+i+1;
        d.d.u_next[id] = d.d.u_tent[id]-(DT / RHO)*d.d.d_p_next_d_x[id];
        d.d.v_next[id] = d.d.v_tent[id]-(DT / RHO)*d.d.d_p_next_d_y[id];
    }
}

Real out_u[NX][NY], out_v[NX][NY], out_p[NX][NY];

void save_npz(data_t d) {
    d.copy_to_host();
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int idx = (1 + j) * NXG + 1 + i;
            out_u[i][j] = d.h.u[idx];
            out_v[i][j] = d.h.v[idx];
            out_p[i][j] = d.h.p[idx];
        }
    }
    char fName[50];
    std::stringstream ss;
    ss << "res_" << std::setfill('0') << std::setw(10) << d.g.step;

    strcpy(fName, ss.str().c_str());
    strcat(fName, ".npz");

    cnpy::npz_save(fName, "U", &(out_u[0][0]), {NX, NY}, "w");
    cnpy::npz_save(fName, "V", &(out_v[0][0]), {NX, NY}, "a");
    cnpy::npz_save(fName, "P", &(out_p[0][0]), {NX, NY}, "a");

    d.calc_exact();
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int idx = (1 + j) * NXG + 1 + i;
            out_u[i][j] = d.h.u_exact[idx];
            out_v[i][j] = d.h.v_exact[idx];
            out_p[i][j] = d.h.p_exact[idx];
        }
    }
    cnpy::npz_save(fName, "U_exact", &(out_u[0][0]), {NX, NY}, "a");
    cnpy::npz_save(fName, "V_exact", &(out_v[0][0]), {NX, NY}, "a");
    cnpy::npz_save(fName, "P_exact", &(out_p[0][0]), {NX, NY}, "a");

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int idx = (1 + j) * NXG + 1 + i;
            out_u[i][j] = d.h.x[idx];
            out_v[i][j] = d.h.y[idx];
        }
    }
    cnpy::npz_save(fName, "X", &(out_u[0][0]), {NX, NY}, "a");
    cnpy::npz_save(fName, "Y", &(out_v[0][0]), {NX, NY}, "a");
}


int main() {
    data_t data;
    data.mem_alloc();
    data.g.hx = L / (N_POINTS - 1);
    data.g.hy = L / (N_POINTS - 1);
    init<<<grid, threads>>>(data); checkErr(cudaGetLastError()); checkErr(cudaGetLastError());
    fill_bnd<<<grid, threads>>>(data.d.u_prev); checkErr(cudaGetLastError());
    fill_bnd<<<grid, threads>>>(data.d.v_prev); checkErr(cudaGetLastError());
    fill_bnd<<<grid, threads>>>(data.d.p_prev); checkErr(cudaGetLastError());
    save_npz(data);
    Real maximum_possible_DT = 0.5 * data.g.hx * data.g.hx / NU;

    if (DT > STABILITY_SAFETY_FACTOR * maximum_possible_DT) {
        return -1;//throw RuntimeError("Stability is not guarenteed");
    }

    data.g.t = 0.;
    data.g.step = 0;
    time_t begin, end;
    time(&begin);
    while (data.g.t < MAX_TIME) {
        data.g.t += DT;
        ++data.g.step;
        central_difference_x<<<grid, threads>>>(data, data.d.d_u_prev_d_x, data.d.u_prev); checkErr(cudaGetLastError());
        central_difference_y<<<grid, threads>>>(data, data.d.d_u_prev_d_y, data.d.u_prev); checkErr(cudaGetLastError());
        central_difference_x<<<grid, threads>>>(data, data.d.d_v_prev_d_x, data.d.v_prev); checkErr(cudaGetLastError());
        central_difference_y<<<grid, threads>>>(data, data.d.d_v_prev_d_y, data.d.v_prev); checkErr(cudaGetLastError());
        laplace<<<grid, threads>>>(data, data.d.laplace_u_prev, data.d.u_prev); checkErr(cudaGetLastError());
        laplace<<<grid, threads>>>(data, data.d.laplace_v_prev, data.d.v_prev); checkErr(cudaGetLastError());
        calc_tent_vel<<<grid, threads>>>(data); checkErr(cudaGetLastError());
        fill_bnd<<<grid, threads>>>(data.d.u_tent); checkErr(cudaGetLastError());
        fill_bnd<<<grid, threads>>>(data.d.v_tent); checkErr(cudaGetLastError());

        central_difference_x<<<grid, threads>>>(data, data.d.d_u_tent_d_x, data.d.u_tent); checkErr(cudaGetLastError());
        central_difference_y<<<grid, threads>>>(data, data.d.d_v_tent_d_y, data.d.v_tent); checkErr(cudaGetLastError());

        calc_p_rhs<<<grid, threads>>>(data); checkErr(cudaGetLastError());

        for (Int ip = 0; ip < N_PRESSURE_POISSON_ITERATIONS; ip++) {
            calc_p_next<<<grid, threads>>>(data); checkErr(cudaGetLastError());
            fill_bnd<<<grid, threads>>>(data.d.p_next); checkErr(cudaGetLastError());
            //cudaMemcpy(data.d.p_prev, data.d.p_next, sizeof(Real) * NXG * NYG, cudaMemcpyDeviceToDevice); checkErr(cudaGetLastError());
            copy_dev_to_dev(data.d.p_prev, data.d.p_next); checkErr(cudaGetLastError());
        }

        central_difference_x<<<grid, threads>>>(data, data.d.d_p_next_d_x, data.d.p_next); checkErr(cudaGetLastError());
        central_difference_y<<<grid, threads>>>(data, data.d.d_p_next_d_y, data.d.p_next); checkErr(cudaGetLastError());

        calc_next_vel<<<grid, threads>>>(data); checkErr(cudaGetLastError());

        fill_bnd<<<grid, threads>>>(data.d.u_next); checkErr(cudaGetLastError());
        fill_bnd<<<grid, threads>>>(data.d.v_next); checkErr(cudaGetLastError());
        copy_dev_to_dev(data.d.u_prev, data.d.u_next); checkErr(cudaGetLastError());
        copy_dev_to_dev(data.d.v_prev, data.d.v_next); checkErr(cudaGetLastError());
        copy_dev_to_dev(data.d.p_prev, data.d.p_next); checkErr(cudaGetLastError());
        cudaDeviceSynchronize();


        if (data.g.step % LOG_STEP == 0) {
            time(&end);
            time_t elapsed = end - begin;
            printf("%d: Time elapsed for %d steps: %ld seconds.\n", data.g.step, LOG_STEP, elapsed);
            time(&begin);
        }

        if (data.g.step % SAVE_STEP == 0) {
            save_npz(data);
        }
    }

    save_npz(data);

    Real u_err_l1 = 0.;
    Real v_err_l1 = 0.;
    Real p_err_l1 = 0.;
    Real u_err_l2 = 0.;
    Real v_err_l2 = 0.;
    Real p_err_l2 = 0.;

    for (Int i = 0; i < NX-1; i++) {
        for (Int j = 0; j < NY-1; j++) {
            Int id = (j+1)*NXG+i+1;
            u_err_l1 += fabs(data.h.u[id]-data.h.u_exact[id]);
            v_err_l1 += fabs(data.h.v[id]-data.h.v_exact[id]);
            p_err_l1 += fabs(data.h.p[id]-data.h.p_exact[id]);
            u_err_l2 += (data.h.u[id]-data.h.u_exact[id])*(data.h.u[id]-data.h.u_exact[id]);
            v_err_l2 += (data.h.v[id]-data.h.v_exact[id])*(data.h.v[id]-data.h.v_exact[id]);
            p_err_l2 += (data.h.p[id]-data.h.p_exact[id])*(data.h.p[id]-data.h.p_exact[id]);
        }
    }
    u_err_l1 *= data.g.hx*data.g.hy;
    v_err_l1 *= data.g.hx*data.g.hy;
    p_err_l1 *= data.g.hx*data.g.hy;

    u_err_l2 *= data.g.hx*data.g.hy;
    v_err_l2 *= data.g.hx*data.g.hy;
    p_err_l2 *= data.g.hx*data.g.hy;

    u_err_l2 = sqrt(u_err_l2);
    v_err_l2 = sqrt(v_err_l2);
    p_err_l2 = sqrt(p_err_l2);

    std::cout << "N = " << N_POINTS-1 << std::endl;
    std::cout << "||u_err||_L1 = " << u_err_l1 << std::endl;
    std::cout << "||v_err||_L1 = " << v_err_l1 << std::endl;
    std::cout << "||p_err||_L1 = " << p_err_l1 << std::endl;
    std::cout << "||u_err||_L2 = " << u_err_l2 << std::endl;
    std::cout << "||v_err||_L2 = " << v_err_l2 << std::endl;
    std::cout << "||p_err||_L2 = " << p_err_l2 << std::endl;

    return 0;
}
