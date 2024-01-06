/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-1
 * Description: Computation of a matrix C = Kronecker_prod(A, B.T)
 *              where A and B are matrices of dimension (m, n) and
 *              the output is of the dimension (m * n, m * n). 
 * Note: All lines marked in --> should be replaced with code. 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
using namespace std;

ofstream outfile; // The handle for printing the output

__global__ void per_row_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned Aid = ceil((float) (id/m));
    unsigned Bid = id % m;
    
    if (id < m*m)
    {
        // For elements in the row selected for the thread(matrix A)
        for(int i=0; i<n; i++)
        {
            //For elements in the row selected for the thread(matrix B)
            for(int j =0; j<n; j++)
            {
                C[n*m*n*Aid + m*i + Bid + m*n*j] = A[Aid*n + i]*B[Bid*n + j];
            }

        }
    }

}

__global__ void per_column_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....
    unsigned id = blockIdx.x * blockDim.y * blockDim.x +  blockDim.y * threadIdx.x + threadIdx.y;
    unsigned Aid = ceil((float)(id/n));
    unsigned Bid = id % n;

    if (id < n*n)
    {
        for(int i =0; i<m; i++)
        {
            for(int j=0; j<m; j++)
            {
                C[n*m*n*i + m*Aid + m*n*Bid + j] = A[i*n + Aid] * B[j*n + Bid];
            }
        }
    }
}

__global__ void per_element_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....
    unsigned blockid = gridDim.y * blockIdx.x + blockIdx.y;
    unsigned id = blockid * blockDim.y * blockDim.x + blockDim.y*threadIdx.x + threadIdx.y; 
    
    if(id < n*m*n*m)
    {
        unsigned aid = ceil((float) (id / (m*n)));
        unsigned bid = id % (m*n);
        unsigned cid = (bid / n) + (bid%n)*m*n + (aid%n)*m + (aid/n)*m*n*n;
        C[cid] = A[aid]*B[bid];

    }
    
}

/**
 * Prints any 1D array in the form of a matrix
 **/
void printMatrix(long int *arr, long int rows, long int cols, char* filename){
    outfile.open(filename);
    for(long int i = 0; i < rows; i++){
        for(long int j = 0; j < cols; j++){
            outfile<<arr[i * cols + j]<<" ";
        }
        outfile<<"\n";
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    long int m,n;	
    cin>>m>>n;	

    // Host_arrays 
    long int *h_a,*h_b,*h_c;

    // Device arrays 
    long int *d_a,*d_b,*d_c;
	
    // Allocating space for the host_arrays 
    h_a = (long int *) malloc(m * n * sizeof(long int));
    h_b = (long int *) malloc(m * n * sizeof(long int));	
    h_c = (long int *) malloc(m * m * n * n * sizeof(long int));	

    // Allocating memory for the device arrays 
    // --> Allocate memory for A on device 
    cudaMalloc(&d_a, m*n*sizeof(long int));
    // --> Allocate memory for B on device 
    cudaMalloc(&d_b, m*n*sizeof(long int));
    // --> Allocate memory for C on device 
    cudaMalloc(&d_c, m*m*n*n*sizeof(long int));

    // Read the input matrix A 
    for(long int i = 0; i < m * n; i++) {
        cin>>h_a[i];
    }

    //Read the input matrix B 
    for(long int i = 0; i < m * n; i++) {
        cin>>h_b[i];
    }

    // Transfer the input host arrays to the device 
    // --> Copy A from Host to Device
    cudaMemcpy(d_a, h_a, m*n*sizeof(long int), cudaMemcpyHostToDevice);
    // --> Copy B from Host to Device 
    cudaMemcpy(d_b, h_b, m*n*sizeof(long int), cudaMemcpyHostToDevice);

    long int gridDimx, gridDimy;
    
    // Launch the kernels
    /**
     * Kernel 1 - per_row_AB_kernel
     * To be launched with 1D grid, 1D block
     * Each thread should process a complete row of A, B
     **/

    // --> Set the launch configuration 
    int blocksize = 1024;
    int nblocks = ceil((float)((m*m)/blocksize));

    double starttime = rtclock();  

    // --> Launch the kernel 
    per_row_AB_kernel<<<nblocks, blocksize>>>(d_a, d_b, d_c, m, n);
    cudaDeviceSynchronize();                                                           

    double endtime = rtclock(); 
	printtime("GPU Kernel-1 time: ", starttime, endtime);  

    // --> Copy C from Device to Host 
    cudaMemcpy(h_c, d_c, m * m * n * n * sizeof(long int), cudaMemcpyDeviceToHost);
    printMatrix(h_c, m * n, m * n,"kernel1.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(long int));

    /**
     * Kernel 2 - per_column_AB_kernel
     * To be launched with 1D grid, 2D block
     * Each thread should process a complete column of  A, B
     **/
    
    // --> Set the launch configuration 
    nblocks = ceil((float) (n*n/blocksize));
    dim3 block(32, 32, 1);
    starttime = rtclock(); 

    // --> Launch the kernel 
    per_column_AB_kernel<<<nblocks, block>>>(d_a, d_b, d_c, m, n);
    cudaDeviceSynchronize(); 

    endtime = rtclock(); 
  	printtime("GPU Kernel-2 time: ", starttime, endtime);  

    // --> Copy C from Device to Host
    cudaMemcpy(h_c, d_c, m * n * m * n * sizeof(long int), cudaMemcpyDeviceToHost);
    printMatrix(h_c, m * n, m * n,"kernel2.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(long int));

    /**
     * Kernel 3 - per_element_kernel
     * To be launched with 2D grid, 2D block
     * Each thread should process one element of the output 
     **/
    gridDimx = ceil(float(n * n) / 16);
    gridDimy = ceil(float(m * m) / 64);
    dim3 grid3(gridDimx,gridDimy,1);
    dim3 block3(64,16,1);

    starttime = rtclock();  

    // --> Launch the kernel 
    per_element_kernel<<<grid3, block3>>>(d_a, d_b, d_c, m, n);
    cudaDeviceSynchronize();                                                              

    endtime = rtclock();  
	printtime("GPU Kernel-3 time: ", starttime, endtime);  

    // --> Copy C from Device to Host
    cudaMemcpy(h_c, d_c, m * n * n * m * sizeof(long int), cudaMemcpyDeviceToHost);
    printMatrix(h_c, m * n, m * n,"kernel3.txt");

    return 0;
}