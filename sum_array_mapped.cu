#include<stdio.h>

#define N 100
#define threadsPerBlock 64
#define numBlocks 2

__global__ void sum(int* a, int* b, int* c){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < N) c[tid] = a[tid] + b[tid];
}

int main(void){
    int *a, *b, *c;
    int *a_dev, *b_dev, *c_dev;

    //Mapped memory
    cudaHostAlloc((void**)&a, N * sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc((void**)&b, N * sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc((void**)&c, N * sizeof(int), cudaHostAllocMapped);

    for(int i=0; i<N; i++){
        a[i] = i;
        b[i] = i * i;
    }

    cudaHostGetDevicePointer(&a_dev, a, 0);
    cudaHostGetDevicePointer(&b_dev, b, 0);
    cudaHostGetDevicePointer(&c_dev, c, 0);

    sum<<<numBlocks, threadsPerBlock>>>(a_dev, b_dev, c_dev);

    for(int i=0; i<N; i++){
        printf("%d + %d = %d \n", a[i], b[i], c[i]);
    }

    cudaFreeHost(a_dev);
    cudaFreeHost(b_dev);
    cudaFreeHost(c_dev);

    return 0;

}