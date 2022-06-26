/************************************************************
* Author   : Harish R
* Created  : June 26 2022
* Modified : June 26 2022
* Purpose  : Memory sharing between CPU & GPU
*************************************************************/
#include <stdio.h>
#include <cuda.h>

#define N 100

__global__ void dkernel(int a[]){
    a[threadIdx.x] = threadIdx.x*threadIdx.x;
}
int main(){
    int a[N], *da;

    cudaMalloc(&da, N*sizeof(int));
    dkernel<<<1,N>>>(da);
    cudaMemcpy(a, da, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    for(int i=0; i<N; i++){
        printf("%d\n", a[i]);
    }
    cudaDeviceSynchronize();
    return 0;
}
