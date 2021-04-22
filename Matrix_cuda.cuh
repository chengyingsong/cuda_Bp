#ifndef MATRIX_CUDA_H
#define MATRIX_CUDA_H
#include"cuda_runtime.h"
#include"Matrix.cuh"


__global__ void matMulKernel(const double *A,const double *B,double *C,int M,int N,int K);
__global__ void matAddKernel(const double *A,const double *B,double *C,int M,int N);
__global__ void matSubKernel(const double *A,const double *B,double *C,int M,int N);
__global__ void matTimesKernel(const double a,const double *B,double *C,int M,int N);
__global__ void matDotKernel(const double *A,const double *B,double *C,int M,int N);
__global__ void matsubsKernel(const double a,const double *B,double *C,int M,int N);
__global__ void matDivKernel(const double a,const double *B,double *C,int M,int N);
__global__ void matTransKernel(const double *B,double *C,int M,int N);
//__global__ void matinitKernel();  暂时没必要优化，时间很快
__global__ void matSigmoidKernel(double *C,int M,int N); 
__global__ void matReluKernel(double *C,int M,int N);
__global__ void matExpKernel(double *C,int M,int N);
  

#endif