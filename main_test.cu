#include<cstdio>
#include"Matrix.cuh"
#include"Matrix_cuda.cuh"
using namespace std;



int main(){
   Matrix A(3,4,NULL);
   Matrix B(1,4,NULL);

   for(int i=0;i<3;i++)
       for(int j=0;j<4;j++)
          A.data[i*4+j] = i *4 +j;
   
   A.print();
   A = A +A;
   cudaDeviceSynchronize();
   A.print();
   return 0;
}