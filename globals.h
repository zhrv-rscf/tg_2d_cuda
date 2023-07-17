#ifndef RSCF_TG_2D_CUDA_GLOBALS_H
#define RSCF_TG_2D_CUDA_GLOBALS_H

#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include <cstring>
#include <cmath>
#include <sstream>
#include <iomanip>

#define CUDA_DEBUG

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

const int SAVE_STEP = 1000;
const int LOG_STEP = 1000;


#define BLOCK_SIZE 4

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

static dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
static dim3 grid(NX / threads.x + 1, NY / threads.y + 1);

static Real *u, *v, *p, *u_star, *v_star, *fu, *fv, *gu, *gv, *rhs_p, *delta_p, *u_old, *v_old, *p_old;
static Real *u_h, *v_h, *p_h;

static Real p_out[NX][NY];
static Real u_out[NX][NY];
static Real v_out[NX][NY];


#endif //RSCF_TG_2D_CUDA_GLOBALS_H
