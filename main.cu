#include "globals.h"
#include "advance.h"
#include "output.h"
#include "init.h"



template<typename T>
__host__
Real *mallocFieldsOnDevice(int nx, int ny) {
    T *c;
    cudaError_t result;
    result = cudaMalloc(&c, sizeof(T) * nx * ny);
    checkErr(result);
    return c;
}







int main() {
//    cudaError_t result;
    u_h = new Real[NXG * NYG];
    v_h = new Real[NXG * NYG];
    p_h = new Real[NXG * NYG];
    u = mallocFieldsOnDevice<Real>(NXG, NYG);
    v = mallocFieldsOnDevice<Real>(NXG, NYG);
    p = mallocFieldsOnDevice<Real>(NXG, NYG);
    u_old = mallocFieldsOnDevice<Real>(NXG, NYG);
    v_old = mallocFieldsOnDevice<Real>(NXG, NYG);
    p_old = mallocFieldsOnDevice<Real>(NXG, NYG);
    fu = mallocFieldsOnDevice<Real>(NX + 1, NY);
    fv = mallocFieldsOnDevice<Real>(NX + 1, NY);
    gu = mallocFieldsOnDevice<Real>(NX, NY + 1);
    gv = mallocFieldsOnDevice<Real>(NX, NY + 1);

    rhs_p = mallocFieldsOnDevice<Real>(NXG, NYG);
    delta_p = mallocFieldsOnDevice<Real>(NXG, NYG);

    u_out = new Real[NX * NY];
    v_out = new Real[NX * NY];
    p_out = new Real[NX * NY];

    init<<<grid, threads>>>(u, v, p);
    checkErr(cudaGetLastError());

    cudaMemcpy(u_h, u, sizeof(Real) * NXG * NYG, cudaMemcpyDeviceToHost);
    cudaMemcpy(v_h, u, sizeof(Real) * NXG * NYG, cudaMemcpyDeviceToHost);
    cudaMemcpy(p_h, u, sizeof(Real) * NXG * NYG, cudaMemcpyDeviceToHost);
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
        compute_single_step();
        compute_single_step();
        compute_substep2_val<<<grid, threads>>>(u, v, p, u_old, v_old, p_old);
        compute_single_step();
        compute_substep3_val<<<grid, threads>>>(u, v, p, u_old, v_old, p_old);

        cudaDeviceSynchronize();

        if (step % LOG_STEP == 0) {
            time(&end);
            time_t elapsed = end - begin;
            printf("%d: Time measured for %d steps: %ld seconds.\n", LOG_STEP, step, elapsed);
            time(&begin);
        }
        if (step % SAVE_STEP == 0) {
            cudaMemcpy(u_h, u, sizeof(Real) * NXG * NYG, cudaMemcpyDeviceToHost);
            cudaMemcpy(v_h, u, sizeof(Real) * NXG * NYG, cudaMemcpyDeviceToHost);
            cudaMemcpy(p_h, u, sizeof(Real) * NXG * NYG, cudaMemcpyDeviceToHost);
            save(u_h, v_h, p_h, step);
        }
    }

    cudaMemcpy(u_h, u, sizeof(Real) * NXG * NYG, cudaMemcpyDeviceToHost);
    cudaMemcpy(v_h, u, sizeof(Real) * NXG * NYG, cudaMemcpyDeviceToHost);
    cudaMemcpy(p_h, u, sizeof(Real) * NXG * NYG, cudaMemcpyDeviceToHost);
    save(u_h, v_h, p_h, step);

    return 0;
}
