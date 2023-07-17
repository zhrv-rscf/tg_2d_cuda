cmake_minimum_required(VERSION 3.25)
project(rscf_tg_2d_cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_subdirectory(cnpy)

add_executable(rscf_tg_2d_cuda main.cu reconstr.cu reconstr.h globals.h fluxes.cu fluxes.h advance.cu advance.h bndcond.cu bndcond.h output.cu output.h init.cu init.h)

target_link_libraries(rscf_tg_2d_cuda z cnpy)

set_target_properties(rscf_tg_2d_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
