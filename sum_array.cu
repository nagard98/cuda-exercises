#include<stdio.h>

#define N 100
#define threadsPerBlock 64
#define numBlocks 2

__global__ void sum(int* a, int* b, int* c){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < N) c[tid] = a[tid] + b[tid];
}

int main(void){
    int a[N], b[N], c[N];
    int *a_dev, *b_dev, *c_dev;

    //Pageable memory
    //cudaMalloc((void**)&a_dev, N * sizeof(int));
    //cudaMalloc((void**)&b_dev, N * sizeof(int));
    //cudaMalloc((void**)&c_dev, N * sizeof(int));

    //Pinned memory
    cudaHostAlloc((void**)&a_dev, N * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&b_dev, N * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&c_dev, N * sizeof(int), cudaHostAllocDefault);

    for(int i=0; i<N; i++){
        a[i] = i;
        b[i] = i * i;
    }

    cudaMemcpy(a_dev, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, N * sizeof(int), cudaMemcpyHostToDevice);

    sum<<<numBlocks, threadsPerBlock>>>(a_dev, b_dev, c_dev);
    
    cudaMemcpy(c, c_dev, N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i<N; i++){
        printf("%d + %d = %d \n", a[i], b[i], c[i]);
    }

    cudaFreeHost(a_dev);
    cudaFreeHost(b_dev);
    cudaFreeHost(c_dev);

    return 0;

}