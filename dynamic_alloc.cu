#include <stdio.h>

 __global__ void mallocThreadTest(){
    size_t size = 1024;
    int* data = (int*)malloc(size);
    memset(data, 0, size);
    printf("Thread %d has got pointer %p \n", threadIdx.x, data);
    free(data);
 }

 __global__ void mallocBlockTest(){
    __shared__ int* data;

    if(threadIdx.x == 0){
        //For each thread we allocate 64Bytes
        size_t size = blockDim.x * 64;
        data = (int*)malloc(size);
    }

    __syncthreads();

   if(data == NULL) return;

    for(int i=0; i<64; i++){
      data[i* blockDim.x + threadIdx.x] = threadIdx.x;
    }

    __syncthreads();
    if( threadIdx.x == 0) free(data);
 }

 int main(void){
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    mallocBlockTest<<<4,128>>>();
    cudaDeviceSynchronize();

    return 0;
 }