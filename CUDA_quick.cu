%%writefile quick.cu
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
#define INSERTION_SORT  32
#define CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         exit(err); \
     } \
} while(0)

//単純選択ソート
template<typename _arraytype>
__device__ 
  void selection_sort(_arraytype *data, int left, int right)
{
    for (long long i = left ; i <= right ; ++i)
    {
        unsigned min_val = data[i];
        long long min_idx = i;
        for (long long j = i+1 ; j <= right ; ++j)
        {
            unsigned val_j = data[j];

            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }
        if (i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

template<typename _arraytype>
__global__ 
  void quicksort(_arraytype *data,long long left,long long right,int depth)
{
    //選択ソートに切り替える
    if (depth >= 20 || right-left<=INSERTION_SORT)
    {
        selection_sort(data,left,right);
        return;
    }

    _arraytype *l=data+left;
    _arraytype *r=data+right;
    _arraytype pivot=data[(left+right)/2];
    while (l<=r)
    {
        _arraytype lval=*l;
        _arraytype rval=*r;
        while (lval<pivot)
        {
            l++;
            lval=*l;
        }
        while (rval>pivot)
        {
            r--;
            rval=*r;
        }
        if (l<=r)
        {
            *l++=rval;
            *r--=lval;
        }
    }
    long long nright=r-data;
    long long nleft=l-data;

    if (left<(r-data))
    {
        //ストリームの指定
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        //非同期ストリームの破棄
        cudaStreamDestroy(s);
    }

    if ((l-data) < right)
    {
        //ストリームの指定
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        //非同期ストリームの破棄
        cudaStreamDestroy(s1);
    }
}


int main(){
    ofstream outputfile("quick_result.txt");
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
            int size=pow(2,i);
            A=(int*)malloc(sizeof(int)*size);//配列の領域確保
            int i=0;
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
            quicksort <<< 1,1 >>>(DA,0,size-1,0);
            //デバイスメモリからホストメモリ
            CUDA_SAFE_CALL(cudaMemcpy(A,DA,sizeof(int)*size,cudaMemcpyDefault));
            stop = clock();
            //デバイスメモリ解放
            cudaFree(DA);
            for (i=1;i<size;i++){
                if (A[i-1]>A[i]){
                    cout << "______________FALSE_________" << endl;
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
