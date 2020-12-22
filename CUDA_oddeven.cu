%%writefile oddeven.cu
#include<iostream>
#include<fstream>
#include<stdio.h>
#include<string>
#include<stdlib.h>
#include<time.h>
using namespace std;
#define BS 1024
//cudaMemcpyが正常に行われたかのチェック
#define CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         exit(err); \
     } \
} while(0)

template<typename _arraytype>
__global__
    void OddSort(_arraytype *array,long long size)
{   
    long i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i*2+2>=size)return;
    if (array[i*2+1]>array[i*2+2]){
        _arraytype dummy=array[i*2+1];
        array[i*2+1]=array[i*2+2];
        array[i*2+2]=dummy;
    }
    return;
}

template<typename _arraytype>
__global__
    void EvenSort(_arraytype *array,long long size)
{   
    long i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i*2+1>=size)return;
    if (array[i*2]>array[i*2+1]){
        _arraytype dummy=array[i*2];
        array[i*2]=array[i*2+1];
        array[i*2+1]=dummy;
    }
    return;
}

int main(){
    ofstream outputfile("oddeven_result.txt");
    for (int i=10;i<=20;i++){
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
            long long size=pow(2,i);
            A=(int*)malloc(sizeof(int)*size);//配列の領域確保
            long long i=0;
            if (ifs.fail()) {
                cout << "Failed to open file." << endl;
                return -1;
            }
            while (getline(ifs, num)) {
                A[i]=atoi(num.c_str());
                i++;
            }

            //GPUメモリ確保
            cudaMalloc((void**)&DA,sizeof(int)*size);
            //ソート実行
            start = clock();
            //ホストメモリからデバイスメモリ
            CUDA_SAFE_CALL(cudaMemcpy(DA,A,sizeof(int)*size,cudaMemcpyDefault));
            for (i=0;i<=size;i++){
                if (i%2){
                    OddSort<<<size/BS,BS>>>(DA,size);
                }else {
                    EvenSort<<<size/BS,BS>>>(DA,size);
                }
            }
            //デバイスメモリからホストメモリ
            CUDA_SAFE_CALL(cudaMemcpy(A,DA,sizeof(int)*size,cudaMemcpyDefault));
            
            stop = clock();
            //デバイスメモリ解放
            cudaFree(DA);
            for (i=1;i<size;i++){
                if (A[i-1]>A[i]){
                    cout << "_____FALSE____";
                    break;
                }
            }
            free(A);
            outputfile << (long double)(stop-start) / CLOCKS_PER_SEC << " " << size << '\n';
        }
    }
    outputfile.close();
    return 0;
}   
