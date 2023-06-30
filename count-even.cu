#include<stdio.h>

#define N 500
#define threadsPerBlock 64

__global__ void countEven(int* a, int* count){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    int ct = 0;

    __shared__ int tmp[threadsPerBlock / 32];
    int lane = threadIdx.x % 32;
    int wid =  threadIdx.x / 32;

    if(tid < N){
        if(a[tid] % 2 == 0){
            ct = 1;
            for(int offset = 16; offset > 0; offset /= 2){
                ct += __shfl_down_sync(__activemask(), ct, offset);
            }
        }
    }

    if(lane == 0) tmp[wid] = ct;
    __syncthreads();

    if(wid == 0){
        ct = lane < (threadsPerBlock / 32) ? tmp[lane] : 0;
        for(int offset = 16; offset > 0; offset /= 2){
            ct += __shfl_down_sync(__activemask(), ct, offset);
        }
    }

    __syncthreads();
    if(threadIdx.x == 0) atomicAdd(count, ct);
}

int main(void){
    int a[N], count = 0;
    int *a_dev, *count_dev;

    //Pinned memory
    cudaHostAlloc((void**)&a_dev, N * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&count_dev, sizeof(int), cudaHostAllocDefault);

    for(int i=0; i<N; i++){
        a[i] = i;
    }

    cudaMemcpy(a_dev, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(count_dev, &count, sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = ceil(N / threadsPerBlock);
    countEven<<<numBlocks, threadsPerBlock>>>(a_dev, count_dev);
    
    cudaMemcpy(&count, count_dev , sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d \n", count);

    cudaFreeHost(a_dev);
    cudaFreeHost(count_dev );

    return 0;

}