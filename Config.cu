#include<cstdio>
#include<vector>
#include "Matrix.cuh"
using namespace std;

vector<int> layers = { 784,512,10 };
int layer_count = layers.size();
double eta = 0.0001;          //ѧϰ��
int batch_size = 500;
int epoch_size = 10;
int train_len = 6000;
int test_len = 1000;


//����ȫ��������
vector<Matrix> W;        //Ȩ�ؾ���
vector<Matrix> Bias;     //��Ԫbias
vector<Matrix> Avg_W(layer_count-1);    //Ȩ�ؾ���
vector<Matrix> Avg_Bias(layer_count-1); //��Ԫbias
vector<double> epoch_acc;
