/*********************************************************************************
* Author   : Harish R
* Created  : June 26 2022
* Modified : June 26 2022
* Purpose  : Kernels to compute C = tran(A+B) - tran(B-A) (equivalent to 2*tran(A)
* Inputs   : ABsolute path to the text file in the PATH variable
***********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCKSIZE 1024
#define GRIDSIZE 65535

#define PATH "input3.txt" //Enter the absolute path to the input file

__global__ void per_row_column_kernel(unsigned *matrix,
                                   unsigned *result,
                                   unsigned long long rows,
                                   unsigned long long cols){
    unsigned long long id = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned long long ii = id/cols;
    if(id<(rows*cols))
	for(unsigned long long jj=0; jj<cols; ++jj)
      result[jj*rows + ii] = 2*matrix[ii*cols + jj];  
}

__global__ void per_column_row_kernel(unsigned *matrix,
                                   unsigned *result,
                                   unsigned long long rows,
                                   unsigned long long cols){
	unsigned long long id = blockIdx.x*blockDim.x*blockDim.y + threadIdx.x*blockDim.y + threadIdx.y;
    unsigned long long ii = id/rows;	
    if(id<(rows*cols))
		for(unsigned long long jj=0; jj<rows; ++jj)
    	  result[jj + ii*rows] = 2*matrix[ii + jj*cols];  
}

__global__ void per_element_kernel(unsigned *matrix,
                                   unsigned *result,
                                   unsigned long long rows,
                                   unsigned long long cols){
	unsigned long long id = (blockIdx.x*gridDim.x + blockIdx.y)*blockDim.x*blockDim.y + threadIdx.x*blockDim.y + threadIdx.y;
    unsigned long long ii = id/cols;
    unsigned long long jj = id%cols;
	    
	if(id<(rows*cols))
		result[jj*rows + ii] = 2*matrix[ii*cols + jj];
}

void prettyprint(unsigned *matrix, unsigned long long rows, unsigned long long cols){
    for(int i=0; i<rows; ++i){
      for(int j=0; j<cols; ++j)
        printf("%u ", matrix[i*cols + j]);   
      printf("\n");
    }
    printf("************\n");
}

int main(){
    unsigned *matrix, *hmatrix, *result, *hresult;
    unsigned long long N, M, nblocks;
    FILE *fptr;
   	fptr = fopen(PATH, "r");
    if(fptr == NULL){
        printf("File not found!");
        return 0;
    }
    fscanf(fptr, "%llu", &N);
    fscanf(fptr, "%llu", &M);
    //printf("%llu\t%llu\n", N, M);
    
    cudaMalloc(&matrix, N*M* sizeof(unsigned));
    cudaMalloc(&result, N*M*sizeof(unsigned));
    hmatrix = (unsigned *)malloc(N*M*sizeof(unsigned));
    hresult = (unsigned *)malloc(N*M*sizeof(unsigned));
    
    for(int i=0; i<N; ++i)
      for(int j=0; j<M; ++j)
        fscanf(fptr, "%u", &hmatrix[i*M+j]);
    fclose(fptr);

    //prettyprint(hmatrix, N, M);
    
    cudaMemcpy(matrix, hmatrix, N*M*sizeof(unsigned), cudaMemcpyHostToDevice);
	
	/******PER_ROW_COLUMN_KERNEL********/   
    nblocks = ceil((float)N*M/BLOCKSIZE);
    dim3 grid1(nblocks, 1, 1);
	dim3 block1(BLOCKSIZE, 1, 1);
	per_row_column_kernel<<<grid1, block1>>>(matrix, result, N, M);
    cudaMemcpy(hresult, result, N*M*sizeof(unsigned), cudaMemcpyDeviceToHost);
	printf("Result of per_row_coulmn_kernel \n");
	prettyprint(hresult, M, N);   	

	/******PER_COLUMN_ROW_KERNEL********/
    nblocks = ceil((float)N*M/BLOCKSIZE);
    dim3 grid2(nblocks, 1, 1);
	dim3 block2(BLOCKSIZE/2, 2, 1); //Since total threads per block cannot exceed 1024.
	per_column_row_kernel<<<grid2, block2>>>(matrix, result, N, M);
    cudaMemcpy(hresult, result, N*M*sizeof(unsigned), cudaMemcpyDeviceToHost);
	printf("Result of per_column_row_kernel \n");
    prettyprint(hresult, M, N);
    
	/******PER_ELEMENT_KERNEL********/
    nblocks = ceil((float)N*M/BLOCKSIZE);
    nblocks = ceil((float)nblocks/GRIDSIZE);
    dim3 grid3(GRIDSIZE, nblocks, 1); 
	dim3 block3(BLOCKSIZE/2, 2, 1); //Since total threads per block cannot exceed 1024.
	per_column_row_kernel<<<grid3, block3>>>(matrix, result, N, M);
    cudaMemcpy(hresult, result, N*M*sizeof(unsigned), cudaMemcpyDeviceToHost);
	printf("Result of per_element_kernel \n");
    prettyprint(hresult, M, N);
    
    return 0;
}
