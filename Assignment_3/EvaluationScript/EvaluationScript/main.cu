/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/
// Kernel to get the number of layer 0 vertices.
__global__ void Layer0(int *csr, int *off, int *apr, int *layers, int V)
{
    unsigned idx =blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < V)
    {
        if(apr[idx] == 0)
        {
            atomicAdd(&layers[1], 1);
        }

        if(idx == 0)
        {
            atomicSub(&layers[0], 1);
            atomicSub(&layers[1], 1);
        }
    }
}

//Kernel which calculates the number of vertices in each layer except for layer 0
__global__ void Layers(int *csr, int *off, int *apr, int *layers, int l)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < layers[l -1])
    {
        for(int i =off[idx]; i < off[idx + 1]; i++)
        {
            atomicMax(&layers[l], csr[i]);
        }
    }
}

// Kernel for Activation Rule
__global__ void Act(int *apr, int *aid, unsigned int *act, int *layers, int l)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= layers[l] + 1 && idx < layers[l+1] + 1)
    {
        if(aid[idx] >= apr[idx])
        {
            atomicInc(&act[idx], 1);
        }
    }
}

//Kernel for Deactivation Rule
__global__ void DeAct(int *apr,int *aid,  unsigned int *act, int*layers , int l)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx > layers[l] + 1 && idx < layers[l+1])
    {
        if(act[idx - 1] ==0 && act[idx + 1] == 0)
        {
            atomicCAS(&act[idx], 1, 0);
        }
    }
}

//Kernel to calculate the AIDs for the next layer
__global__ void AIDs(int *csr, int *off, int *apr, int*aid, unsigned int *act, int*active, int *layers, int l)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= layers[l] + 1 && idx < layers[l+1] + 1)
    {
        if(act[idx] == 1)
        {
            atomicAdd(&active[l], 1);
            for(int i=off[idx]; i<off[idx + 1]; i++)
            {
                atomicAdd(&aid[csr[i]], 1);
            }
        }
    }
}
/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
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
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
	cudaMalloc(&d_activeVertex, L*sizeof(int));


/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/
//creating variables for storing number of vertices in each layer
int *dlayers;
unsigned int *d_act;

//Allocating space and initializing
cudaMalloc(&dlayers, (L+1)*sizeof(int));
cudaMemset(dlayers, 0, (L+1)*sizeof(int));

cudaMalloc(&d_act, V * sizeof(unsigned int));
cudaMemset(d_act, 0, V * sizeof(unsigned int));

cudaMemset(d_aid, 0 , V*sizeof(int));
cudaMemset(d_activeVertex, 0, L*sizeof(int));

//N Blocks and Blocksize
int blocksize = 1024;
int nblocks = ceil((float) V/ blocksize);

//Launching kernels
Layer0<<<nblocks, blocksize>>>(d_csrList, d_offset, d_apr, dlayers, V);
//Going Layer by Layer
for(int l = 0; l<L; l++)
{
    if(l>=1)
    {
        Layers<<<nblocks, blocksize>>>(d_csrList, d_offset, d_apr, dlayers, l + 1);
    }
    Act<<<nblocks, blocksize>>>(d_apr, d_aid, d_act, dlayers, l);
    DeAct<<<nblocks, blocksize>>>(d_apr, d_aid, d_act, dlayers, l);
    AIDs<<<nblocks, blocksize>>>(d_csrList, d_offset, d_apr, d_aid, d_act, d_activeVertex, dlayers, l);
}

cudaMemcpy(h_activeVertex, d_activeVertex, L*sizeof(int), cudaMemcpyDeviceToHost);

/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
