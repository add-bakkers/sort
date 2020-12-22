%%writefile bitonic.cu
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include<iostream>
#include<fstream>
#include<string>
#include<stdlib.h>
using namespace std;

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 1024
//cudaMemcpyが正常に行われたかのチェック
#define CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         exit(err); \
     } \
} while(0)

__constant__
int arraysize;  

template<typename _arraytype>
__global__ 
  void bitonic_sort_step(_arraytype *dev_values,int j,int k)
{
    unsigned int i, ixj; 
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i^j;
    if ((ixj)>i) {
        if ((i&k)==0) {
            if (dev_values[i]>dev_values[ixj]) {
                    _arraytype temp = dev_values[i];
                    dev_values[i] = dev_values[ixj];
                    dev_values[ixj] = temp;
            }
        }
        if ((i&k)!=0) {
            if (dev_values[i]<dev_values[ixj]) {
                    _arraytype temp = dev_values[i];
                    dev_values[i] = dev_values[ixj];
                    dev_values[ixj] = temp;
            }
        }
    }
}


template<typename _arraytype>
    void bitonic_sort(_arraytype *array,int _arraysize)
{
    for (int k = 2; k <= _arraysize; k <<= 1) {
        for (int j=k>>1; j>0; j=j>>1) {
            bitonic_sort_step<<<_arraysize/THREADS, THREADS>>>(array, j, k);
        }
    }
}

int main(){
    ofstream outputfile("bitonic_result.txt");
    for (int i=10;i<=17;i++){
        for (int j=0;j<10;j++){
            string s="sample2^";
            s+=to_string(i);
            s+="-";
            s+=to_string(j);
            s+=".txt";
            ifstream ifs(s);
            int *A;
            int *DA;
            clock_t start, stop;
            string num;
            int size=pow(2,i);
            A=(int*)malloc(sizeof(int)*size);//配列の領域確保
            int k=0;
            if (ifs.fail()) {
                cout << "Failed to open file." << endl;
                return -1;
            }
            while (getline(ifs, num)) {
                A[k]=atoi(num.c_str());
                k++;
            }
            cudaMemcpyToSymbol(
                arraysize,
                &size,
                sizeof(int)*1,
                0,
                cudaMemcpyHostToDevice);
            //GPUメモリ確保
            cudaMalloc((void**)&DA,sizeof(int)*size);
            //ソート実行
            start = clock();
            //ホストメモリからデバイスメモリ
            CUDA_SAFE_CALL(cudaMemcpy(DA,A,sizeof(int)*size,cudaMemcpyDefault));
            bitonic_sort(DA,size);
            //デバイスメモリからホストメモリ
            CUDA_SAFE_CALL(cudaMemcpy(A,DA,sizeof(int)*size,cudaMemcpyDefault));
            stop = clock();
            //デバイスメモリ解放
            cudaFree(DA);
            //check
            for (k=1;k<size;k++){
                if (A[k-1]>A[k]){
                    cout << "______________FALSE_________" << endl;
                    break;
                }
            }
            outputfile << (long double)(stop-start) / CLOCKS_PER_SEC << " " << size << '\n';
        }
        cout << i << endl;
    }
    outputfile.close();
    return 0;
}   
