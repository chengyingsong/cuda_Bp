#include<cstdio>
#include<vector>
#include "Matrix.cuh"
using namespace std;

vector<int> layers = { 784,512,10 };
int layer_count = layers.size();
double eta = 0.001;          //学习率
int batch_size = 500;
int epoch_size = 10;
int train_len = 6000;
int test_len = 1000;


//三层全连接网络
vector<Matrix> W;        //权重矩阵
vector<Matrix> Bias;     //单元bias
vector<Matrix> Avg_W(layer_count-1);    //权重矩阵
vector<Matrix> Avg_Bias(layer_count-1); //单元bias
vector<double> epoch_acc;
