#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

# define BLOCKSIZE 32

__global__ void MAT_ADD(int *A, int *B, int *O, int rows, int cols)
{
	__shared__ int Tile_A[BLOCKSIZE][BLOCKSIZE + 1];
	__shared__ int Tile_B[BLOCKSIZE][BLOCKSIZE + 1];

	int i = blockIdx.x * BLOCKSIZE + threadIdx.x;
	int j = blockIdx.y * BLOCKSIZE + threadIdx.y;

	if(j < rows && i < cols)
	{
		int id = i + j * cols;
		Tile_A[threadIdx.y][threadIdx.x] = A[id];
		Tile_B[threadIdx.y][threadIdx.x] = B[id];
	}
	__syncthreads();

	i = blockIdx.x * BLOCKSIZE + threadIdx.x;
	j = blockIdx.y * BLOCKSIZE + threadIdx.y;

	if (j < rows && i < cols)
	{
		int idx = i + j * cols;
		O[idx] = Tile_A[threadIdx.y][threadIdx.x] + Tile_B[threadIdx.y][threadIdx.x];
	}
}

__global__ void MAT_T(int *A, int *AT, int cols, int rows)
{
	__shared__ int Tile_A[BLOCKSIZE][BLOCKSIZE + 1];

	unsigned int i = blockIdx.x * BLOCKSIZE + threadIdx.x;
	unsigned int j = blockIdx.y * BLOCKSIZE + threadIdx.y;

	if ((j < rows) && (i < cols)){
		unsigned int id = j * cols + i;
		Tile_A[threadIdx.y][threadIdx.x] = A[id];
	}
	__syncthreads();

	i = blockIdx.y * BLOCKSIZE + threadIdx.x;
	j = blockIdx.x * BLOCKSIZE + threadIdx.y;

	if((i < rows) && (j < cols))
	{
		unsigned int idx = j * rows + i;
		AT[idx] = Tile_A[threadIdx.x][threadIdx.y];
	}

}

__global__ void MAT_MUL(int *A, int *B, int *O, int rA, int cA, int cB)
{
	__shared__ int Tile_A[BLOCKSIZE][BLOCKSIZE];
	__shared__ int Tile_B[BLOCKSIZE][BLOCKSIZE];

	int c = threadIdx.x + BLOCKSIZE * blockIdx.x;
	int r = threadIdx.y + BLOCKSIZE * blockIdx.y;

	int k, tile_id, out = 0;

	for(tile_id = 0; tile_id < (cA + BLOCKSIZE - 1)/BLOCKSIZE; tile_id++)
	{
		if((r < rA) && (tile_id * BLOCKSIZE + threadIdx.x < cA))
		{
			Tile_A[threadIdx.y][threadIdx.x] = A[r*cA + tile_id * BLOCKSIZE + threadIdx.x];
		}
		else
		{
			Tile_A[threadIdx.y][threadIdx.x] = 0;
		}

		if((c < cB) && (tile_id * BLOCKSIZE + threadIdx.y < cA))
		{
			Tile_B[threadIdx.y][threadIdx.x] = B[(tile_id * BLOCKSIZE + threadIdx.y) * cB + c];
		}
		else
		{
			Tile_B[threadIdx.y][threadIdx.x] = 0;
		}
		__syncthreads();

		for(k = 0; k <BLOCKSIZE; k++)
		{
			out += Tile_A[threadIdx.y][k] * Tile_B[k][threadIdx.x];
		}
		__syncthreads();
	}

	if(r < rA && c < cB)
	{
		O[r * cB + c] = out;
	}
}

// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
	int *d_DT, *d_CDT, *d_AB;
	cudaMalloc(&d_DT, q * r * sizeof(int));
	cudaMalloc(&d_CDT, p * r * sizeof(int));
	cudaMalloc(&d_AB, p * r * sizeof(int));

	// D Transpose 
	int grid1x = ceil((float)q / BLOCKSIZE);
	int grid1y = ceil((float)r / BLOCKSIZE);
	dim3 grid1(grid1x, grid1y);
	dim3 block1(BLOCKSIZE, BLOCKSIZE);
	MAT_T<<<grid1, block1>>>(d_matrixD, d_DT, q, r);
	cudaDeviceSynchronize();

	// C @ DT
	int grid2x = ceil((float) r / BLOCKSIZE);
	int grid2y = ceil((float) p / BLOCKSIZE);
	dim3 grid2(grid2x, grid2y);
	dim3 block2(BLOCKSIZE, BLOCKSIZE);
	MAT_MUL<<<grid2, block2>>>(d_matrixC, d_DT, d_CDT, p, q, r);
	cudaDeviceSynchronize();

	// A @ B 
	int grid3x = ceil((float) r / BLOCKSIZE);
	int grid3y = ceil((float) p / BLOCKSIZE);
	dim3 grid3(grid3x, grid3y);
	dim3 block3(BLOCKSIZE, BLOCKSIZE);
	MAT_MUL<<<grid3, block3>>>(d_matrixA, d_matrixB, d_AB, p, q, r);
	cudaDeviceSynchronize();

	// E = A@B + C@DT
	int grid4x = ceil((float) r / BLOCKSIZE);
	int grid4y = ceil((float) p / BLOCKSIZE);
	dim3 grid4(grid4x, grid4y);
	dim3 block4(BLOCKSIZE, BLOCKSIZE);
	MAT_ADD<<<grid4, block4>>>(d_AB, d_CDT, d_matrixE, p, r);
	cudaDeviceSynchronize();

	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	
