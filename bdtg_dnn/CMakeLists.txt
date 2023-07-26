cmake_minimum_required(VERSION 3.16.3)
project(bdtg_dnn)

find_package(ROOT REQUIRED)
#include_directories(${ROOT_INCLUDE_DIRS})
#link_directories(${ROOT_LIBRARY_DIR})

option(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-10")
option(CMAKE_CXX_COMPILER "/usr/bin/gcc++-10")
option(CMAKE_C_COMPILER "/usr/bin/gcc-10")
option(tmva-gpu ON)

set(TMVA_GPU_HEADERS TMVA/DNN/Architectures/Cuda.h)
                  #  TMVA/DNN/Architectures/Cuda/CudaBuffers.h
                         #  TMVA/DNN/Architectures/Cuda/CudaMatrix.h
                                #  TMVA/DNN/Architectures/Cuda/CudaTensor.h

find_package(CUDAToolkit)
include("${ROOT_USE_FILE}")
separate_arguments(ROOT_EXE_LINKER_FLAGS)

SET(CMAKE_CXX_FLAGS "-lTMVA -Dcuda=ON -Dcudnn=ON -Dtmva-gpu=ON -Dall=ON ${CMAKE_CXX_FLAGS}")

add_executable ( run_bulk run_bulk.cpp )
target_link_libraries ( run_bulk PUBLIC ${ROOT_LIBRARIES}
                                                  ${ROOT_EXE_LINKER_FLAGS}
                                                  PRIVATE CUDA::cudart)

add_executable ( run_single run_single.cpp )
target_link_libraries ( run_single PUBLIC ${ROOT_LIBRARIES}
                                                  ${ROOT_EXE_LINKER_FLAGS}
                                                  PRIVATE CUDA::cudart)

