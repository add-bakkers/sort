%%writefile merge.cu
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
    void Merge(_arraytype *array,_arraytype *subarray,long long phase,long long size)
{
    long long i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i*phase*2>size)return;
    long long l=i*phase*2;
    long long r=l+phase;
 
    for (long long j=0;j<phase*2 and j<size-(i*phase*2);j++){
        if (l>=i*phase*2+phase){
            subarray[i*phase*2+j]=array[r];
            r++;
        }else if(r>=(i+1)*phase*2 or r>=size){
            subarray[i*phase*2+j]=array[l];
            l++;
        }else if(array[l]<array[r]){
            subarray[i*phase*2+j]=array[l];
            l++;
        }else{
            subarray[i*phase*2+j]=array[r];
            r++;
        }
    }

    for (long long j=i*phase*2;j<(i+1)*phase*2;j++){
        array[j]=subarray[j];
    }
    __syncthreads();
    return;
}

template<typename _arraytype>
    void MergeSort(_arraytype *array,_arraytype *subarray,long long size)
{   
    for (long long i=0;(1<<i)<size;i++){
        Merge<<<size/(1<<i)/BS+1,BS>>>(array,subarray,(1<<i),size);
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
            int *DAsub;
            clock_t start, stop;
            string num;
            long long size=pow(2,i);
            A=(int*)malloc(sizeof(int)*size);//配列の領域確保
            if (ifs.fail()) {
                cout << "Failed to open file." << endl;
                return -1;
            }
            int k=0;
            while (getline(ifs, num)) {
                A[k]=atoi(num.c_str());
                k++;
            }

            //GPUメモリ確保
            cudaMalloc((void**)&DA,sizeof(int)*size);
            cudaMalloc((void**)&DAsub,sizeof(int)*size);
            //ソート実行
            start = clock();
            //ホストメモリからデバイスメモリ
            CUDA_SAFE_CALL(cudaMemcpy(DA,A,sizeof(int)*size,cudaMemcpyDefault));

            MergeSort(DA,DAsub,size);
            //デバイスメモリからホストメモリ
            CUDA_SAFE_CALL(cudaMemcpy(A,DA,sizeof(int)*size,cudaMemcpyDefault));
            
            stop = clock();
            //デバイスメモリ解放
            cudaFree(DA);
            cudaFree(DAsub);
            for (k=1;k<size;k++){
                if (A[k-1]>A[k]){
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
