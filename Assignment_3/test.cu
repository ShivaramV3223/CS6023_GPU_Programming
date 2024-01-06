#include <stdio.h>
#include <cuda.h>

__device__ unsigned int syncCounter = 0;

__device__ void GlobalBarrier()
{
    __syncthreads();
    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        atomicMin(&syncCounter, 0);
    }
    __syncthreads();

    if(threadIdx.x == 0)
    {
        atomicInc(&syncCounter, gridDim.x);
    }
    __syncthreads();

    while (syncCounter < gridDim.x)
    {}
    __syncthreads();
    
}

__global__ void PreProcess(int*csr, int*off, int *apr, int*layers, int L)
{
    int li = 0;
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(apr[idx] == 0)
    {
        atomicAdd(&layers[1], 1);
    }

    GlobalBarrier();
    
    for(int l = 1; l < L + 1; l++)
    {
        if(idx < li + layers[l - 1])
        {
            for(int i = off[idx]; i < off[idx+ 1]; i++)
            {
                atomicMax(&layers[l], csr[i]);
            }
        }
        GlobalBarrier();
        
    }

    if(idx ==0)
    {
        atomicSub(&layers[0], 1);
        atomicSub(&layers[1], 1);
    }
}

__global__ void AID(int*csr, int*off, int*apr, int*layers, int*aid, unsigned int*act,  int L, int V)
{
   unsigned tdx = blockDim.x * blockIdx.x + threadIdx.x;

   for(int l = 0; l < L; l++)
   {
        if(tdx >= layers[l] + 1 &&  tdx < layers[l+1] +1)
        {
            if(aid[tdx] >= apr[tdx])
            {
                atomicInc(&act[tdx], 1);
            }
            GlobalBarrier();

            if(tdx > layers[l] + 1 && tdx < layers[l+1])
            {
                if(act[tdx - 1]  == 0 && act[tdx + 1] == 0)
                {
                    atomicCAS(&act[tdx], 1, 0);
                }
            }
            GlobalBarrier();

            if(act[tdx] == 1)
            {
                for(int i =off[tdx]; i < off[tdx +1]; i++)
                {
                    atomicAdd(&aid[csr[i]], 1);
                }
            }
            GlobalBarrier();
        }
        GlobalBarrier();
   }
}

__global__ void Counts(int *layers, unsigned int *act, int * active, int L)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

    for(int l=0; l< L; l++)
    {
        if(idx >= layers[l] + 1 && idx < layers[l + 1] + 1)
        {
            if(act[idx] == 1)
            {
                atomicAdd(&active[l], 1);
            }
        }
        GlobalBarrier();
    }
}

int main(void)
{
    int csr[32] = {5, 6, 5, 6 ,7, 7 ,7 ,8 , 9, 8, 9, 10, 10, 11, 10, 11, 11, 12, 13, 13, 14, 14, 15, 15, 16, 15, 16, 17, 17, 18, 19, 19};
    int off[21] = {0,2, 5, 6, 9, 11, 12, 14, 16, 19, 20, 21, 23, 25, 27, 28, 31, 32, 32, 32, 32};
    int layers[6] = {0};
    int aid[20] = {0};
    int apr[20] = {0, 0, 0, 0, 0, 2, 4, 2, 2, 2, 2, 3, 1, 2, 2, 2, 3, 1, 1, 1};
    int active[5] = {0};
    unsigned int act[20] = {0};
    int *d_off, *d_csr, *d_apr, *d_layers, *d_aid, *d_active;
    unsigned int*d_act;

    cudaMalloc(&d_off, 21 * sizeof(int));
    cudaMemcpy(d_off, off, 21*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_apr, 20*sizeof(int));
    cudaMemcpy(d_apr, apr, 20*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_csr, 32*sizeof(int));
    cudaMemcpy(d_csr, csr, 32*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_layers, 6 * sizeof(int));
    cudaMemcpy(d_layers, layers, 6*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_active, 5 * sizeof(int));
    cudaMemcpy(d_active, active, 5*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_act, 20*sizeof(unsigned int));
    cudaMalloc(&d_aid, 20*sizeof(unsigned int));



    PreProcess<<<1,20>>>(d_csr, d_off, d_apr, d_layers, 5);
    AID<<<1, 20>>>(d_csr, d_off, d_apr, d_layers, d_aid, d_act, 5, 20);
    Counts<<<1, 20>>>(d_layers, d_act, d_active, 5);

    cudaMemcpy(layers, d_layers, 6*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(act, d_act, 20*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(aid, d_aid, 20 *sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(active, d_active, 5*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i< 20; i++)
    {
        printf("%d) %d %d\n", i, act[i], aid[i]);
    }

    for(int i=0; i<6; i++)
    {
        printf("Layer %d -> %d\n", i, layers[i]);
    }

    for(int i=0; i<5; i++)
    {
        printf("Layer %d --> %d\n", i, active[i]);
    }
    
}