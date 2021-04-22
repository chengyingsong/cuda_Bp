#ifndef MATRIX_H
#define MATRIX_H

#include<cstdio>
#include<stdlib.h>
#include <random>
#include <cmath>
#include<vector>
#include<fstream>
#include"cuda_runtime.h"
using namespace std;

class Matrix{
public:
    double* data;
    int shape0;
    int shape1;
    Matrix(int shape0,int shape1,double* d);
    Matrix(int shape0,int shape1);
    Matrix();
    Matrix dot(const Matrix &b);
    Matrix transpose();
    void init(double std);
    void assign(double* data,int size);
    void print();
    void save(fstream &file);
    void load(fstream &file);
    vector<int> argmax(int dim);
};

Matrix operator-(const Matrix &m1, const Matrix &m2);
Matrix operator+(const Matrix &m1, const Matrix &m2);
Matrix operator*(const Matrix &m1, const Matrix &m2);
Matrix operator*(double a, const Matrix& m2);
Matrix operator/(const Matrix& m2,double a);
Matrix operator-(double a,const Matrix& m2);
int getblocksize(int M);
#endif