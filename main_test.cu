#include<cstdio>
#include<string>
#include<fstream>

using namespace std;


int main(){
   string s = "data/train-labels.idx1-ubyte";
   ifstream file(s, ios::binary);
   if(file.is_open())
      printf("Succ!\n");
    else printf("wrong!\n");
   return 0;
}