#include<cstdio>
#include<stdlib.h>
#include"assert.h"
#include<fstream>
#include"cuda_runtime.h"
#include"Matrix_cuda.cuh"
#include"Matrix.cuh"
//TODO:实现矩阵的切片和赋值初始化，实现批次下降


Matrix::Matrix(int shape0,int shape1,double* d=NULL){
    //建立一个矩阵，并在共享内存申请空间。
    this->shape0 = shape0;
    this->shape1 = shape1;
    this->data = d;
    int nBytes = shape0*shape1*sizeof(double);
    cudaMallocManaged((void**)&this->data,nBytes);
}


Matrix::Matrix(int shape0,int shape1){
    //在CPU中申请空间
    this->shape0 = shape0;
    this->shape1 = shape1;
    int nBytes = shape0 * shape1 * sizeof(double);
    this->data = (double*)malloc(nBytes);
}

Matrix::Matrix(){this->shape0 =0;this->shape1=0;}


int getblocksize(int M){
    //根据数组维度M和N，确定网格中核比,取2的幂次，接近的比
    assert(M>0);
    double x = sqrt(M);
    int two[] = {1,2,4,8,16,32,64,128,256,512};
    for(int i=0;i<10;i++)
    {
        if(x < two[i])
           return two[i-1];
    } 
    return 1024;
}

Matrix Matrix::dot(const Matrix &b){
    assert(shape1 == b.shape0);
    Matrix An(shape0,b.shape1,NULL);

    dim3 blocksize(getblocksize(shape0),getblocksize(b.shape1));
    dim3 gridsize((shape0+blocksize.x-1)/blocksize.x,(b.shape1+blocksize.y-1)/blocksize.y);
    matMulKernel<< <gridsize,blocksize>> >(this->data,b.data,An.data,shape0,shape1,b.shape1);
    
    return An;
}


Matrix Matrix::transpose(){
    Matrix An(shape1,shape0,NULL);

    dim3 blocksize(getblocksize(shape1),getblocksize(shape0));
    dim3 gridsize((shape1+blocksize.x-1)/blocksize.x,(shape0+blocksize.y-1)/blocksize.y);
    matTransKernel<< <gridsize,blocksize>> >(this->data,An.data,shape1,shape0);
    
    return An;
}

Matrix operator+(const Matrix &a, const Matrix &b){
    assert(a.shape0 == b.shape0 && a.shape1 == b.shape1);
    Matrix An(a.shape0,a.shape1,NULL);
    
    dim3 blocksize(getblocksize(a.shape0),getblocksize(a.shape1));
    dim3 gridsize((a.shape0+blocksize.x-1)/blocksize.x,(a.shape1+blocksize.y-1)/blocksize.y);
    matAddKernel<< <gridsize,blocksize>> >(a.data,b.data,An.data,a.shape0,a.shape1);

    return An;
}

Matrix operator-(const Matrix &a, const Matrix &b){
    assert(a.shape0 == b.shape0 && a.shape1 == b.shape1);
    Matrix An(a.shape0,a.shape1,NULL);
    
    dim3 blocksize(getblocksize(a.shape0),getblocksize(a.shape1));
    dim3 gridsize((a.shape0+blocksize.x-1)/blocksize.x,(a.shape1+blocksize.y-1)/blocksize.y);
    matSubKernel<< <gridsize,blocksize>> >(a.data,b.data,An.data,a.shape0,a.shape1);

    return An;
}


Matrix operator*(const Matrix &a, const Matrix &b){
    assert(a.shape0 == b.shape0 && a.shape1 == b.shape1);
    Matrix An(a.shape0,a.shape1,NULL);
    
    dim3 blocksize(getblocksize(a.shape0),getblocksize(a.shape1));
    dim3 gridsize((a.shape0+blocksize.x-1)/blocksize.x,(a.shape1+blocksize.y-1)/blocksize.y);
    matDotKernel<< <gridsize,blocksize>> >(a.data,b.data,An.data,a.shape0,a.shape1);

    return An;
}


Matrix operator*(double a, const Matrix& m2){
   //广播乘
   Matrix An(m2.shape0,m2.shape1,NULL);
   dim3 blocksize(getblocksize(m2.shape0),getblocksize(m2.shape1));
   dim3 gridsize((m2.shape0+blocksize.x-1)/blocksize.x,(m2.shape1+blocksize.y-1)/blocksize.y);
   matTimesKernel<< <gridsize,blocksize>> >(a,m2.data,An.data,m2.shape0,m2.shape1);

   return An;
}


Matrix operator-(double a, const Matrix& m2){
    //广播减
    Matrix An(m2.shape0,m2.shape1,NULL);
    dim3 blocksize(getblocksize(m2.shape0),getblocksize(m2.shape1));
    dim3 gridsize((m2.shape0+blocksize.x-1)/blocksize.x,(m2.shape1+blocksize.y-1)/blocksize.y);
    matsubsKernel<< <gridsize,blocksize>> >(a,m2.data,An.data,m2.shape0,m2.shape1);
 
    return An;
 }


 Matrix operator/(const Matrix& m2,double a){
    //广播除
    Matrix An(m2.shape0,m2.shape1,NULL);
    dim3 blocksize(getblocksize(m2.shape0),getblocksize(m2.shape1));
    dim3 gridsize((m2.shape0+blocksize.x-1)/blocksize.x,(m2.shape1+blocksize.y-1)/blocksize.y);
    matDivKernel<< <gridsize,blocksize>> >(a,m2.data,An.data,m2.shape0,m2.shape1);
 
    return An;
 }


void Matrix::assign(double* data,int size)
{
    //单个数据的赋值
    for (int i = 0; i < size; i++)
        this->data[i] = data[i];
}

vector<int> Matrix::argmax(int dim)
{
    assert(dim == 0 || dim == 1);
    vector<int> An;
    if (dim == 0)
    {
        for (int i = 0; i < shape0; i++)
        {
            int max_arg = 0;
            for (int j = 1; j < shape1; j++)
            {
                if (data[i*shape1+j] > data[i*shape1+max_arg]) //i行
                    max_arg = j;
            }
            An.push_back(max_arg);
        }
    }
    else
    {
        for (int i = 0; i < shape1; i++)
        {
            int max_arg = 0;
            for (int j = 1; j < shape0; j++)
            {
                if (data[j*shape1+i] > data[max_arg*shape1+i]) //i列
                    max_arg = j;
            }
            An.push_back(max_arg);
        }
    }
    return An;
}


void Matrix::init(double std)
{
    //只运行一次，暂时不优化了，随机初始化
    int seed = 3; //随机种子
    default_random_engine gen(seed);
    normal_distribution<double> dis(0, std);
    //printf("%d %d\n", this->shape0, this->shape1);
    for (int i = 0; i < this->shape0; i++)
        for (int j = 0; j < this->shape1; j++)
            this->data[i*shape1+j] = dis(gen);
}


void Matrix::print(){
    for(int i=0;i<shape0;i++)
    {
        for(int j=0;j<shape1;j++)
           printf("%.2lf ",data[i*shape1+j]);
        printf("\n");
    }
}

void Matrix::save(fstream &file) {
    assert(file.is_open() == true);
    for (int i = 0; i < shape0; i++)
    {
        for (int j = 0; j < shape1; j++)
            file << data[i*shape1 + j] << '\t';
        file << endl;
    }
}


void Matrix::load(fstream &file) {
    assert(file.is_open() == true);
    for (int i = 0; i < shape0; i++)
    {
        for (int j = 0; j < shape1; j++)
            file >> data[i*shape1+j];
       }
}