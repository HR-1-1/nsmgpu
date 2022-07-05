/*********************************************************************************
* Author   : Harish R
* Created  : July 03 2022
* Modified : July 04 2022
* Inputs   : path to the text file 
***********************************************************************************/
#include<iostream>
#include<cuda.h>


using namespace std;

#define TILE_SIZE 32 

void prettyprint(int *matrix, int rows, int cols){
    for(int i=0; i<rows; ++i){
      for(int j=0; j<cols; ++j)
        printf("%u ", matrix[i*cols + j]);   
      printf("\n");
    }
    printf("************\n");
}

__global__ void kernel_matrix_multiplication(int* a, int* b, int* c, int a_rows, int a_columns, int b_columns){	
	__shared__ int shared_a_tile[TILE_SIZE][TILE_SIZE];
	__shared__ int shared_b_tile[TILE_SIZE][TILE_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;
	int row = blockIdx.y * TILE_SIZE + threadIdx.y;

	//check if thread directly maps to the dimensions of the resulting matrix
	
		int result = 0;
		int k;
		int phase;
		
		for (phase = 0; phase < (a_columns+TILE_SIZE-1)/TILE_SIZE; phase++)
		{	
			if ((row < a_rows) && (phase*TILE_SIZE+tx < a_columns))
			{
			shared_a_tile[ty][tx] = a[row * a_columns + phase * TILE_SIZE + tx];
			}
			else {shared_a_tile[ty][tx] = 0;}
			if( (phase*TILE_SIZE+ty <a_columns) && (col <b_columns))
			{
			shared_b_tile[ty][tx] = b[(phase * TILE_SIZE + ty) * b_columns + col];
			}
			else {shared_b_tile[ty][tx] = 0;}
			__syncthreads();
			
			for (k = 0; k < TILE_SIZE; k++)
					result += (shared_a_tile[ty][k] * shared_b_tile[k][tx]);
			__syncthreads();
		}	

		if(row<a_rows && col <b_columns){
			c[row * b_columns + col] = result;
		}
	
}

__global__ void kernel_matrix_addition(int *matrix_one, int *matrix_two, int *matrix_out, int rows, int columns){
	__shared__ int block_one[TILE_SIZE][TILE_SIZE+1];
	__shared__ int block_two[TILE_SIZE][TILE_SIZE+1];

	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;

	if (x < columns && y < rows)
	{	
		int idx_in = y*columns +x;
		block_one[threadIdx.y][threadIdx.x] = matrix_one[idx_in];
		block_two[threadIdx.y][threadIdx.x] = matrix_two[idx_in];
	}
	__syncthreads();

	x = blockIdx.x * TILE_SIZE + threadIdx.x;
	y = blockIdx.y * TILE_SIZE + threadIdx.y;

	if (x < columns && y < rows)
	{	
		int idx_out = y*columns+x;
		matrix_out[idx_out] = block_one[threadIdx.y][threadIdx.x] +block_two[threadIdx.y][threadIdx.x];
	}
}

__global__ void kernel_matrix_transpose(int *odata, int *idata, int width, int height){
	__shared__ int block[TILE_SIZE][TILE_SIZE+1];
	
	unsigned int xIndex = blockIdx.x * TILE_SIZE + threadIdx.x;
	unsigned int yIndex = blockIdx.y * TILE_SIZE + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	xIndex = blockIdx.y * TILE_SIZE + threadIdx.x;
	yIndex = blockIdx.x * TILE_SIZE + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

// function to compute the output matrix
void compute(int p, int q, int r, int s, int *h_matrixA, int *h_matrixB, int *h_matrixC, int *h_matrixD, int *h_matrixX) {
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixX, *d_matrixAt, *d_matrixBt, *d_matrixCt, *d_matrixY, *d_matrixZ;

	int *h_matrixAt, *h_matrixBt, *h_matrixCt, *h_matrixY, *h_matrixZ;
	h_matrixAt = (int*)malloc(q*p*sizeof(int));
	h_matrixBt = (int*)malloc(q*p*sizeof(int));
	h_matrixCt = (int*)malloc(p*r*sizeof(int));
	h_matrixY = (int*)malloc(p*s*sizeof(int));
	h_matrixZ = (int*)malloc(q*p*sizeof(int));

	// allocate memory...
	cudaMalloc(&d_matrixA, p*q*sizeof(int));
	cudaMalloc(&d_matrixB, p*q*sizeof(int));
	cudaMalloc(&d_matrixC, r*p*sizeof(int));
	cudaMalloc(&d_matrixD, r*s*sizeof(int));

	cudaMalloc(&d_matrixAt, q*p*sizeof(int)); // transpose(A)
	cudaMalloc(&d_matrixBt, q*p*sizeof(int)); // transpose(B)
	cudaMalloc(&d_matrixCt, p*r*sizeof(int)); // transpose(C)
	cudaMalloc(&d_matrixX, q*s*sizeof(int)); // output matrix
	cudaMalloc(&d_matrixY, p*s*sizeof(int)); // intermediate matrix C.T*D
	cudaMalloc(&d_matrixZ, q*p*sizeof(int)); // intermediate matrix A.T+B.T

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p*q*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, p*q*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, r*p*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r*s*sizeof(int), cudaMemcpyHostToDevice);
	
	//Compute A.T
	dim3 grid0(ceil(float(q)/TILE_SIZE), ceil(float(p)/TILE_SIZE), 1);
    dim3 threads0(TILE_SIZE, TILE_SIZE, 1);
	kernel_matrix_transpose<<< grid0, threads0 >>>(d_matrixAt, d_matrixA, q, p);
	cudaDeviceSynchronize();
	cudaMemcpy(h_matrixAt,d_matrixAt,q*p*sizeof(int),cudaMemcpyDeviceToHost);
	
	//Compute B.T
	dim3 grid1(ceil(float(q)/TILE_SIZE), ceil(float(p)/TILE_SIZE), 1);
    dim3 threads1(TILE_SIZE, TILE_SIZE, 1);
	kernel_matrix_transpose<<< grid1, threads1 >>>(d_matrixBt, d_matrixB, q, p);
	cudaDeviceSynchronize();
	cudaMemcpy(h_matrixBt,d_matrixBt,q*p*sizeof(int),cudaMemcpyDeviceToHost);

	//Compute C.T
	dim3 grid2(ceil(float(p)/TILE_SIZE), ceil(float(r)/TILE_SIZE), 1);
    dim3 threads2(TILE_SIZE, TILE_SIZE, 1);
	kernel_matrix_transpose<<< grid2, threads2 >>>(d_matrixCt, d_matrixC, p, r);
	cudaDeviceSynchronize();
	cudaMemcpy(h_matrixCt,d_matrixCt,p*r*sizeof(int),cudaMemcpyDeviceToHost);

	//Compute C.T*D
	dim3 grid3(ceil(float(s)/TILE_SIZE), ceil(float(p)/TILE_SIZE), 1);
	dim3 threads3(TILE_SIZE,TILE_SIZE,1);
	kernel_matrix_multiplication<<<grid3, threads3>>>(d_matrixCt,d_matrixD,d_matrixY,p,r,s);
	cudaDeviceSynchronize();
	cudaMemcpy(h_matrixY,d_matrixY,p*s*sizeof(int),cudaMemcpyDeviceToHost);

	//Compute A.T+B.T
	dim3 grid4(ceil(float(q) / TILE_SIZE), ceil(float(p) / TILE_SIZE), 1);
	dim3 threads4(TILE_SIZE, TILE_SIZE, 1);
	kernel_matrix_addition<<<grid4, threads4>>>(d_matrixAt, d_matrixBt, d_matrixZ, q, p);
	cudaDeviceSynchronize();
	cudaMemcpy(h_matrixZ,d_matrixZ,q*p*sizeof(int),cudaMemcpyDeviceToHost);
	
	//Compute (A.T+B.T)*(C.T*D)
	dim3 grid5(ceil(float(s)/TILE_SIZE),ceil(float(q)/TILE_SIZE));
	dim3 threads5(TILE_SIZE,TILE_SIZE);
	kernel_matrix_multiplication<<<grid5, threads5>>>(d_matrixZ,d_matrixY,d_matrixX,q,p,s);
	cudaDeviceSynchronize();
	cudaMemcpy(h_matrixX, d_matrixX, q*s*sizeof(int), cudaMemcpyDeviceToHost); 
	
	//prettyprint(h_matrixAt, p, q);
	//prettyprint(h_matrixA, q, p);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixX);
	cudaFree(d_matrixY);
	cudaFree(d_matrixZ);
	cudaFree(d_matrixAt);
	cudaFree(d_matrixBt);
	cudaFree(d_matrixCt);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}




int main(int argc, char **argv) {
	// variable declarations
	int p, q, r, s;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixX;

	// get file names from command line
	char *inputFileName = "/content/input1.txt";

	// file pointers
	FILE *inputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}
	
	// read input values
	fscanf(inputFilePtr, "%d %d %d %d", &p, &q, &r, &s);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(p * q * sizeof(int));
	matrixC = (int*) malloc(r * p * sizeof(int));
	matrixD = (int*) malloc(r * s * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, p, q);
	readMatrix(inputFilePtr, matrixC, r, p);
	readMatrix(inputFilePtr, matrixD, r, s);

	// allocate memory for output matrix
	matrixX = (int*) malloc(q * s * sizeof(int)); // q x s

	// call compute function to get the output matrix. it is expected that 
	compute(p, q, r, s, matrixA, matrixB, matrixC, matrixD, matrixX);
	cudaDeviceSynchronize();

	//prettyprint(matrixA,p,q);	
	prettyprint(matrixX,q,s);
	
	// close files
    fclose(inputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixX);

	return 0;
}
