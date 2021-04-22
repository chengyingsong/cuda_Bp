#include"Matrix_cuda.cuh"

__global__ void matMulKernel(const double *A,const double *B,double *C,int M,int N,int K){
    double Cvalue = 0.0;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if(x >= M || y>= K)
       return;
    for (int i = 0; i < N; ++i)
    {
        Cvalue += A[x*N + i]* B[i*K+y];
    }
    C[x*K+y] = Cvalue;
}

__global__ void matAddKernel(const double *A,const double *B,double *C,int M,int N){
    int x = threadIdx.x + blockIdx.x *blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x >= M || y>= N)
       return;
    //printf("x:%d,y:%d %d %d\n",x,y,A[x*N+y],B[x*N+y]);
    C[x*N+y] = A[x*N+y] + B[x*N+y];
}

__global__ void matSubKernel(const double *A,const double *B,double *C,int M,int N){
    int x = threadIdx.x + blockIdx.x *blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x >= M || y>= N)
       return;
    //printf("x:%d,y:%d %d %d\n",x,y,A[x*N+y],B[x*N+y]);
    C[x*N+y] = A[x*N+y] - B[x*N+y];
}

__global__ void matDotKernel(const double *A,const double *B,double *C,int M,int N){
    int x = threadIdx.x + blockIdx.x *blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x >= M || y>= N)
       return;
    //printf("x:%d,y:%d %d %d\n",x,y,A[x*N+y],B[x*N+y]);
    C[x*N+y] = A[x*N+y] * B[x*N+y];
}


__global__ void matTimesKernel(const double a,const double *B,double *C,int M,int N){
    int x = threadIdx.x + blockIdx.x *blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x >= M || y>= N)
       return;
    C[x*N+y] = a *  B[x*N+y];
}

__global__ void matTransKernel(const double *B,double *C,int M,int N){
    int x = threadIdx.x + blockIdx.x *blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x >= M || y>= N)
       return;
    //M是shape1，N是shape0，C[x][y] = B[y][x]
    C[x*N+y] =  B[y*M+x];
}


__global__ void matsubsKernel(const double a,const double *B,double *C,int M,int N){
    int x = threadIdx.x + blockIdx.x *blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x >= M || y>= N)
       return;

    C[x*N+y] = a -  B[x*N+y];
}

__global__ void matExpKernel(double *C,int M,int N){
    int x = threadIdx.x + blockIdx.x *blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x >= M || y>= N)
       return;

    C[x*N+y] = exp(C[x*N+y]);
}

__global__ void matReluKernel(double *B,int M,int N){
    int x = threadIdx.x + blockIdx.x *blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x >= M || y>= N)
       return;
    
    if(B[x*N+y]<0)
        B[x*N+y] = 0;
}

__global__ void matSigmoidKernel(double *B,int M,int N){
    int x = threadIdx.x + blockIdx.x *blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x >= M || y>= N)
       return;
    
    B[x*N+y] = 1 / (1 + exp(B[x*N+y]));
}


__global__ void matDivKernel(const double a,const double *B,double *C,int M,int N){
    int x = threadIdx.x + blockIdx.x *blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x >= M || y>= N)
       return;

    C[x*N+y] =  B[x*N+y] / a;
}