#include "output.h"
#include "cnpy/cnpy.h"


void save_npz(int step) {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int id = j * NX + i;
            int idg = (K_WENO + j) * NXG + K_WENO + i;
            u_out[id] = u_h[idg];
            v_out[id] = v_h[idg];
            p_out[id] = p_h[idg];
        }
    }
    char fName[50];
    std::stringstream ss;
    ss << "tg_2d_inc_" << std::setfill('0') << std::setw(10) << step;

    strcpy(fName, ss.str().c_str());
    strcat(fName, ".npz");

    cnpy::npz_save(fName, "U", u_out, {NX, NY}, "a");
    cnpy::npz_save(fName, "V", v_out, {NX, NY}, "a");
    cnpy::npz_save(fName, "P", p_out, {NX, NY}, "a");

}


void save_vtk(Real *u, Real *v, Real *p, int step) {/*
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


*/}


void save(Real *u, Real *v, Real *p, int step) {
    save_vtk(u, v, p, step);
    save_npz(step);
}


