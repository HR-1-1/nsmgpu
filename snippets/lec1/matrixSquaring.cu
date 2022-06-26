/************************************************************
* Author   : Harish R
* Created  : June 26 2022
* Modified : June 26 2022
* Purpose  : Matrix Squaring
*************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#define N 3

//__global__ void init(unsigned *matrix,
//                     unsigned matrixsize){
//    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
//   unsigned ii = id/matrixsize;
//    unsigned jj = id%matrixsize;
//
//    curandState st;
//    curand_init(clock64(), id, 0, &st);
//    matrix[ii*matrixsize + jj] = (curand_uniform(&st)%10);
//}

__global__ void square(unsigned *matrix,
                       unsigned *result,
                       unsigned matrixsize){
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned ii = id/matrixsize;
    unsigned jj = id%matrixsize;
    result[ii*matrixsize + jj] = 0;
    for(unsigned kk=0; kk<matrixsize; ++kk){
        result[ii*matrixsize + jj] += matrix[ii*matrixsize + kk]*
                                      matrix[kk*matrixsize + jj];
    }
}

void prettyprint(unsigned *matrix, unsigned matrixsize){
    for(int i=0; i<matrixsize; ++i){
      for(int j=0; j<matrixsize; ++j)
        printf("%d ", matrix[i*matrixsize + j]);   
      printf("\n");
    }
    printf("************\n");
}

int main(){
    unsigned int *matrix, *hmatrix, *result, *hresult;
    cudaMalloc(&matrix, N*N* sizeof(unsigned));
    cudaMalloc(&result, N*N*sizeof(unsigned));
    hmatrix = (unsigned *)malloc(N*N*sizeof(unsigned));
    hresult = (unsigned *)malloc(N*N*sizeof(unsigned));

    srand(time(NULL));
    for(int i=0; i<N; ++i)
      for(int j=0; j<N; ++j)
        hmatrix[i*N + j] = (rand()%10);
    prettyprint(hmatrix, N);

    cudaMemcpy(matrix, hmatrix, N*N*sizeof(unsigned), cudaMemcpyHostToDevice);
    square<<<N,N>>>(matrix,
                    result,
                    N);
    cudaMemcpy(hresult, result, N*N*sizeof(unsigned), cudaMemcpyDeviceToHost);
    prettyprint(hresult, N);
    return 0;
}
