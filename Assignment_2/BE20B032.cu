#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;


// write kernels here...

#define T_LENGTH 32

__global__ void Mat_add(int *matrixA, int *matrixB, int *OutMatrix, int RowsAB, int ColumnsAB){

    // declaring shared memory
	__shared__ int A_Tile[T_LENGTH][T_LENGTH + 1];
	__shared__ int B_Tile[T_LENGTH][T_LENGTH + 1];
    // setting x and y values
	int x = blockIdx.x * T_LENGTH + threadIdx.x;
	int y = blockIdx.y * T_LENGTH + threadIdx.y;

    // copying values
	if ( y < RowsAB && x < ColumnsAB )
	{	
		int i = x + y * ColumnsAB;
		A_Tile[threadIdx.y][threadIdx.x] = matrixA[i];
		B_Tile[threadIdx.y][threadIdx.x] = matrixB[i];
	}
    // syncing threads across the block
	__syncthreads();
    // setting x and y values
	x = blockIdx.x * T_LENGTH + threadIdx.x;
	y = blockIdx.y * T_LENGTH + threadIdx.y;

    // adding  
	if ( y < RowsAB && x < ColumnsAB )
	{	
		int out = y * ColumnsAB + x;
		OutMatrix[out] = A_Tile[threadIdx.y][threadIdx.x] + B_Tile[threadIdx.y][threadIdx.x];
	}
}

__global__ void Mat_transpose(int *outdata, int *indata, int matWidth, int matHeight){

    // declaring shared memory
	__shared__ int tile[T_LENGTH][T_LENGTH + 1];
    // setting xindex and yindex
	unsigned int xIndex = blockIdx.x * T_LENGTH + threadIdx.x;
	unsigned int yIndex = blockIdx.y * T_LENGTH + threadIdx.y;

    //copying values
	if( (yIndex < matHeight) && (xIndex < matWidth) )
	{
		unsigned int index_in = yIndex * matWidth + xIndex;
		tile[threadIdx.y][threadIdx.x] = indata[index_in];
	}
    // syncing threads across block
	__syncthreads();
     // setting xindex and yindex
	xIndex = blockIdx.y * T_LENGTH + threadIdx.x;
	yIndex = blockIdx.x * T_LENGTH + threadIdx.y;

    // transposing matrix
	if((xIndex < matHeight) && (yIndex < matWidth))
	{
		unsigned int out = yIndex * matHeight + xIndex;
		outdata[out] = tile[threadIdx.x][threadIdx.y];
	}
}

__global__ void Mat_mul(int* a, int* b, int* c, int rowsA, int columnsA, int columnsB){	
	
    //declaring shared memory
	__shared__ int A_tile[T_LENGTH][T_LENGTH];
	__shared__ int B_tile[T_LENGTH][T_LENGTH];
    // setting column and row values
	int col = threadIdx.x + blockIdx.x * T_LENGTH;
	int row = threadIdx.y + blockIdx.y * T_LENGTH;
    // setting threadx and thready
    int threadx = threadIdx.x;
	int thready = threadIdx.y;
	// initialising variable values
	int value = 0;
	int k;
	int t;
		
	for (t = 0; t < (columnsA + T_LENGTH - 1) / T_LENGTH; t++)
		{	//copying to A tile
			if ( (row < rowsA) && (t * T_LENGTH + threadx < columnsA) ) {
			    A_tile[thready][threadx] = a[row * columnsA + t * T_LENGTH + threadx];
			}
			else {
                A_tile[thready][threadx] = 0;
            }
            //copying to B tile
			if( (t * T_LENGTH+thready < columnsA) && (col < columnsB) ){
    			B_tile[thready][threadx] = b[(t * T_LENGTH + thready) * columnsB + col];
			}
			else
                {
                 B_tile[thready][threadx] = 0;
                }
            // syncing threads
			__syncthreads();
			
			for (k = 0; k < T_LENGTH; k++)
			{
				value += (A_tile[thready][k] * B_tile[k][threadx]);
			}
			__syncthreads();
			
		}	
    // writing values to c matrix
	if(row<rowsA && col <columnsB)
        {
			c[row * columnsB + col] = value;
		}
	
}




// function to compute the output matrix
void compute(int p, int q, int r, int s, int *h_matrixA, int *h_matrixB, int *h_matrixC, int *h_matrixD, int *h_matrixX) {
	// variable declarations...
	int *d_A, *d_B, *d_C, *d_D, *d_X;
    int  *d_Bt, *d_Dt;
    int *prod1, *prod2;
	// allocate memory...

    // on gpu
	cudaMalloc(&d_A, p * q * sizeof(int));
	cudaMalloc(&d_B, q * p * sizeof(int));
	cudaMalloc(&d_C, q * r * sizeof(int));
	cudaMalloc(&d_D, s * r * sizeof(int));
    // c * dt
	cudaMalloc(&prod1, q * s * sizeof(int)); 
    // a + bt
	cudaMalloc(&prod2, p * q * sizeof(int)); 
    // transpose of B
	cudaMalloc(&d_Bt, p * q * sizeof(int)); 
    // transpose of D
	cudaMalloc(&d_Dt, r * s * sizeof(int)); 
    // Final matrix
	cudaMalloc(&d_X, p * s * sizeof(int)); 
    

	// copy the values...
	cudaMemcpy(d_A, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_matrixB, q * p * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_matrixC, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_D, h_matrixD, s * r *sizeof(int), cudaMemcpyHostToDevice);
	

    // call the kernels for doing required computations...

    //Doing b transpose
    int grid1x = ceil(float(p) / T_LENGTH);
    int grid1y = ceil(float(q) / T_LENGTH);
    // setting block and grid dimensions
	dim3 grid1( grid1x, grid1y);
    dim3 block1(T_LENGTH, T_LENGTH);
    //calling the kernel
	Mat_transpose<<< grid1, block1 >>>(d_Bt, d_B, p, q);
	cudaDeviceSynchronize();

    //doing d transpose
    int grid2x = ceil(float(r) / T_LENGTH);
    int grid2y = ceil(float(s) / T_LENGTH);
     // setting block and grid dimensions
	dim3 grid2(grid2x, grid2y);
    dim3 block2(T_LENGTH, T_LENGTH);
    //calling the kernel
	Mat_transpose<<< grid2, block2 >>>(d_Dt, d_D, r, s);
	cudaDeviceSynchronize();

    // multiplying C *dt
    int grid3x = ceil(float(s) / T_LENGTH);
    int grid3y = ceil(float(q) / T_LENGTH);
    // setting block and grid dimensions
	dim3 grid3(grid3x, grid3y);
	dim3 block3(T_LENGTH,T_LENGTH);
    //calling the kernel
	Mat_mul<<<grid3, block3>>>(d_C, d_Dt, prod1, q, r, s);
	cudaDeviceSynchronize();

    //Adding a and bt
    int grid4x = ceil(float(p) / T_LENGTH);
    int grid4y = ceil(float(q) / T_LENGTH);
    // setting block and grid dimensions
	dim3 grid4(grid4x, grid4y);
	dim3 block4(T_LENGTH, T_LENGTH);
    //calling the kernel
	Mat_add<<<grid4, block4>>>(d_A, d_Bt, prod2, p, q);
	cudaDeviceSynchronize();
	
    // Final multiplication
    int grid5x = ceil(float(s) / T_LENGTH);
    int grid5y = ceil(float(p) / T_LENGTH);
    // setting block and grid dimensions
	dim3 grid5(grid5x,grid5y);
	dim3 block5(T_LENGTH,T_LENGTH);
    //calling the kernel
	Mat_mul<<<grid5, block5>>>(prod2, prod1, d_X, p, q, s);
	cudaDeviceSynchronize();

    // copy the result back...
	cudaMemcpy(h_matrixX, d_X, p*s*sizeof(int), cudaMemcpyDeviceToHost); 

	// deallocate the memory...
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_D);
	cudaFree(d_X);
	cudaFree(prod1);
	cudaFree(prod2);
	cudaFree(d_Bt);
	cudaFree(d_Dt);
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
	int p, q, r, s;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixX;
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
	fscanf(inputFilePtr, "%d %d %d %d", &p, &q, &r, &s);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * p * sizeof(int));
	matrixC = (int*) malloc(q * r * sizeof(int));
	matrixD = (int*) malloc(s * r * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, p);
	readMatrix(inputFilePtr, matrixC, q, r);
	readMatrix(inputFilePtr, matrixD, s, r);

	// allocate memory for output matrix
	matrixX = (int*) malloc(p * s * sizeof(int)); // p x s

	// call compute function to get the output matrix. it is expected that 
	// the compute function will store the result in matrixX.
	gettimeofday(&t1, NULL);
	compute(p, q, r, s, matrixA, matrixB, matrixC, matrixD, matrixX);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixX, p, s); // p x s

	// close files
    fclose(inputFilePtr);
    fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixX);

	return 0;
}