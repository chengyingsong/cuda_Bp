#include"model.cuh"
#include"Matrix.cuh"
#include"cuda_runtime.h"
#include"Matrix_cuda.cuh"
#include "assert.h"
#include<string>
#include<fstream>
using namespace std;

extern int layer_count;
extern vector<int> layers;
extern vector<Matrix> W;        //Ȩ�ؾ���
extern vector<Matrix> Bias;     //��Ԫbias
extern vector<Matrix> Avg_W;    //Ȩ�ؾ���
extern vector<Matrix> Avg_Bias; //��Ԫbias

void ReLU(Matrix& a)
{
    dim3 blocksize(getblocksize(a.shape0),getblocksize(a.shape1));
    dim3 gridsize((a.shape0+blocksize.x-1)/blocksize.x,(a.shape1+blocksize.y-1)/blocksize.y);
    matReluKernel<< <gridsize,blocksize>> >(a.data,a.shape0,a.shape1);
}


void sigmoid(Matrix& a) {
    dim3 blocksize(getblocksize(a.shape0),getblocksize(a.shape1));
    dim3 gridsize((a.shape0+blocksize.x-1)/blocksize.x,(a.shape1+blocksize.y-1)/blocksize.y);
    matSigmoidKernel<< <gridsize,blocksize>> >(a.data,a.shape0,a.shape1);
}

void tanh(Matrix& a) {
    a = 2 * a;
    sigmoid(a);
    a = -1 - (-2) * a; //2a-1
}

Matrix tanh_derivative(const Matrix& a) {
    return 1- a * a;

}

Matrix ReLu_derivative(Matrix& a)
{
    Matrix An(a.shape0, a.shape1);
    for (int i = 0; i < a.shape0; i++)
        for (int j = 0; j < a.shape1; j++)
            if (a.data[i*a.shape1+j] > 0)
                An.data[i*a.shape1+j] = 1;
            else
                An.data[i*a.shape1+j] = 0;
    return An;
}

void Softmax(Matrix& a)
{
    //shape (1,10)
    //����exp
    dim3 blocksize(getblocksize(a.shape0),getblocksize(a.shape1));
    dim3 gridsize((a.shape0+blocksize.x-1)/blocksize.x,(a.shape1+blocksize.y-1)/blocksize.y);
    matExpKernel<< <gridsize,blocksize>> >(a.data,a.shape0,a.shape1);
    for (int i = 0; i < a.shape0; i++)
    {
        //print(a);
        double sum_x = 0;
        for (int j = 0; j < a.shape1; j++)
            sum_x += a.data[i*a.shape1+j];
        for (int j = 0; j < a.shape1; j++)
            a.data[i*a.shape1+j] /= sum_x;
    }
}

vector<Matrix> forward(Matrix& sample_x)
{
    //sample_x shape (1,784)
    // W[0] (784,512),W[1]
    vector<Matrix> H;
    H.push_back(sample_x);
    for (int i = 1; i < layer_count - 1; i++)
    { //H[1] = x W + B (1,512)
        Matrix h = H[i - 1].dot(W[i - 1]) + Bias[i - 1];
        tanh(h);
        H.push_back(h);
    }
    //������(1,512),Ȼ��softmax
    Matrix y_hat = H[layer_count - 2].dot(W[layer_count - 2]) + Bias[layer_count - 2];
    //printf("before softmax: ");
    //print(y_hat);
    Softmax(y_hat);
    //printf("after softmax: ");
    //print(y_hat);
    H.push_back(y_hat);
    return H;
}


void backprop(vector<Matrix> H, double y)
{
    //TODO:��ÿһ����ݶȷ���Avg��
    vector<Matrix> delta_error(H.size() - 1); //H.size = 3,y_hat,h1,x
    Matrix Y(1, layers[layers.size() - 1]);  //Y.shape 1,10
    Y.data[int(y)] = 1;
    int index = layer_count - 2; //1
    delta_error[index] = H[index + 1] - Y; //shape (1,10)
    //printf("Predict y_hat: ");
    //print(H[index+1]); 
    Avg_Bias[index] = delta_error[index]; //shape (1,10)
    Avg_W[index] = H[index].transpose().dot(delta_error[index]); //shape(512,10)
    for (int i = index - 1; i >= 0; i--)
    {
        Matrix h_derivative = tanh_derivative(H[i + 1]); //(1,512)
        delta_error[i] = delta_error[i + 1].dot(W[i + 1].transpose()) + h_derivative;
        Avg_Bias[i] = delta_error[i];
        Avg_W[i] = H[i].transpose().dot(delta_error[i]);
    }
    //�ͷ��ڴ�
    for(int i=0;i<delta_error.size();i++)
    {
        cudaFree(delta_error[i].data);
    }
}

void save_model(const char* filename) {
    fstream file(filename, ios::out | ios::trunc); //����д��
    assert(file.is_open() == true);
    //�ȴ���W[0],B[0],W[1],B[0]
    for (int i = 0; i < W.size(); i++)
    {
        //����W[i]
        W[i].save(file);
        file << endl;
        Bias[i].save(file);
    }
    file.close();
}

void load_model(const char* filename) {
    fstream file(filename, ios::in); //��ȡ�ļ�����
    assert(file.is_open() == true);
    //�ȴ���W[0],B[0],W[1],B[0]
    for (int i = 0; i < W.size(); i++)
    {
        //����W[i]
        W[i].load(file);
        Bias[i].load(file);
    }
    file.close();
}
