#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

//*******************************************
// Write down the kernels here
__global__ void prefix_sum(int * arr, int N)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < N)
    {
        for(int off = 1; off < N; off *= 2)
    {
        if(idx >= off)
        {
            int val = (idx) % (2*off);
            if(val >= off)
            {
                arr[idx] += arr[idx - val + off - 1];
            }
        }
    }
    }
    
}

__global__ void Process(int * fac, int * cap, int * req_cen, int * req_fac, int * req_start, int * req_slots, int * tot_reqs, int * succ_reqs, int R, int F)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < F)
    {
      int slots[24] = {0};
    int flag, id;

    for(int i=0; i<R; i++)
    {
        if(req_cen == 0)
           id = 0;
        else
            id = fac[req_cen[i] - 1];

        if(id + req_fac[i] == idx)
        {
            flag = 0;
            for(int j = req_start[i]; j < req_start[i] + req_slots[i]; j++)
            {
                if(slots[j] < cap[idx])
                {
                    slots[j]++;
                }
                else
                {
                    flag = 1;
                    break;
                }
            }

            if(flag == 0)
            {
                atomicAdd(&succ_reqs[req_cen[i]], 1);
            }

            atomicAdd(&tot_reqs[req_cen[i]], 1);
        }
    }
    }
}

//***********************************************


int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 


    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    int k1=0 , k2 = 0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      }
    }

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request
    
    // Allocate memory on CPU 
	int R;
	fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }
		


    //*********************************
    // Call the kernels here
    int *d_fac, *d_cap;
    int *d_req_cen, *d_req_fac, *d_req_start, *d_req_slots, *d_tot_reqs, *d_succ_reqs;

    cudaMalloc(&d_fac, N*sizeof(int));
    cudaMalloc(&d_cap, max_P*N*sizeof(int));
    cudaMalloc(&d_req_cen, R*sizeof(int));
    cudaMalloc(&d_req_fac, R*sizeof(int));
    cudaMalloc(&d_req_start, R*sizeof(int));
    cudaMalloc(&d_req_slots, R*sizeof(int));
    cudaMalloc(&d_tot_reqs, N*sizeof(int));
    cudaMalloc(&d_succ_reqs, N*sizeof(int));

    cudaMemcpy(d_fac, facility, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cap, capacity, max_P*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_cen, req_cen, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_fac, req_fac, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_start, req_start, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_slots, req_slots, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tot_reqs, tot_reqs, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_succ_reqs, succ_reqs, N*sizeof(int), cudaMemcpyHostToDevice);

    int *fac = (int *)malloc(N*sizeof(int));

    prefix_sum<<<1, N>>>(d_fac, N);
    cudaMemcpy(fac, d_fac, N*sizeof(int), cudaMemcpyDeviceToHost);

    int F = fac[N -1];
    int nblocks = ceil((float)F/BLOCKSIZE);

    Process<<<nblocks, BLOCKSIZE>>>(d_fac, d_cap, d_req_cen, d_req_fac, d_req_start, d_req_slots, d_tot_reqs, d_succ_reqs, R, F);
    cudaMemcpy(succ_reqs, d_succ_reqs, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tot_reqs, d_tot_reqs, N*sizeof(int), cudaMemcpyDeviceToHost);    
    //********************************
    // Output
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}