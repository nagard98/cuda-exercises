#include <stdio.h>

#define NUM_THREADS_BLOCK 64
#define INPUT_LEN 65536

__global__ void reduce(int* odata, int* idata){

    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    sdata[tid] = idata[i];
    __syncthreads();

    for(int s = 1; s < blockDim.x; s *= 2){
        if(tid % (s*2) == 0){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }

    if(tid == 0) {
        odata[blockIdx.x] = sdata[0];
        //printf("The block %d produced partial sum = %d \n", blockIdx.x, sdata[0]);
    }

}

__global__ void reduce2(int* odata, int* idata){

    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    sdata[tid] = idata[i];
    __syncthreads();

    for(int s = blockDim.x/2; s > 0 ; s >>= 1){
        if(tid < s){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }

    if(tid == 0) {
        odata[blockIdx.x] = sdata[0];
        //printf("The block %d produced partial sum = %d \n", blockIdx.x, sdata[0]);
    }

}

int main(int argc,char **argv){
    int n_blocks = INPUT_LEN / NUM_THREADS_BLOCK;
    if( INPUT_LEN % NUM_THREADS_BLOCK > 0) n_blocks += 1;

    int *h_odata, *h_idata;
    int *d_odata, *d_idata;

    cudaHostAlloc((void**)&h_odata, n_blocks * sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc((void**)&h_idata, INPUT_LEN * sizeof(int), cudaHostAllocMapped);

    for(int i=0; i<INPUT_LEN; i++) h_idata[i] = 1;
    for(int i=0; i<n_blocks; i++) h_odata[i] = 0;

    cudaHostGetDevicePointer(&d_odata, h_odata, 0);
    cudaHostGetDevicePointer(&d_idata, h_idata, 0);

    while(n_blocks > 0){
        reduce<<<n_blocks , NUM_THREADS_BLOCK, NUM_THREADS_BLOCK * sizeof(int)>>>(d_odata, d_idata);
        cudaDeviceSynchronize();
        cudaMemcpy(d_idata, d_odata, n_blocks * sizeof(int), cudaMemcpyDeviceToDevice);
        //printf("------------------------------------\n");
        if(n_blocks == 1) break;

        int tmp = n_blocks / NUM_THREADS_BLOCK;
        if(n_blocks % NUM_THREADS_BLOCK > 0){
          cudaMemset(d_idata + n_blocks, 0, (NUM_THREADS_BLOCK - (n_blocks % NUM_THREADS_BLOCK)) * sizeof(int)  );
          n_blocks = tmp + 1;
        } 
        else n_blocks = tmp; 
        
    }

    cudaFreeHost(h_odata);
    cudaFreeHost(h_idata);

    return 0;
}