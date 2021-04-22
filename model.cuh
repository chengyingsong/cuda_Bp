#pragma once
#ifndef MODEL_H
#define MODEL_H
#include"Matrix.cuh"
#include<vector>
using namespace std;


void ReLU(Matrix& a);
void sigmoid(Matrix &a);
void tanh(Matrix& a);
Matrix tanh_derivative(const Matrix& a);
Matrix ReLu_derivative(const Matrix& a);
Matrix sigmoid_derivative(const Matrix &a);

void Softmax(Matrix& a);

vector<Matrix> forward(Matrix& sample_x);
void backprop(vector<Matrix> H, double y);

void load_model(const char* filename);
void save_model(const char* filename);

#endif
