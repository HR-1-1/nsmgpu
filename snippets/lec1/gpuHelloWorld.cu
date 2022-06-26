/************************************************************
* Author   : Harish R
* Created  : June 26 2022
* Modified : June 26 2022
* Purpose  : GPU Hello World in Parallel with multiple threads
*************************************************************/
#include <stdio.h>
#include <cuda.h>

__global__ void dkernel(){
    printf("Hello World from %d \n", threadIdx.x);
}

int main(){
    dkernel<<<1,1024>>>();
    cudaDeviceSynchronize();
    return 0;
}
