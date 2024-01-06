#include <stdio.h>
#include <cuda.h>

__global__ void Layer0(int *csr, int*off, int*apr, int *layers, int V)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
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

__global__ void AIDs(int *csr, int *off, int *apr, int*aid, unsigned int *act, int *layers, int l)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= layers[l] + 1 && idx < layers[l+1] + 1)
    {
        if(act[idx] == 1)
        {
            for(int i=off[idx]; i<off[idx + 1]; i++)
            {
                atomicAdd(&aid[csr[i]], 1);
            }
        }
    }
}

__global__ void Count(unsigned int * act, int *active, int*layers, int l)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= layers[l] +1 && idx < layers[l+1] + 1)
    {
        if(act[idx] == 1)
        {
            atomicAdd(&active[l], 1);
        }
    }
}

int main(void)
{
    int csr[32] = {5, 6, 5, 6 ,7, 7 ,7 ,8 , 9, 8, 9, 10, 10, 11, 10, 11, 11, 12, 13, 13, 14, 14, 15, 15, 16, 15, 16, 17, 17, 18, 19, 19};
    int off[21] = {0,2, 5, 6, 9, 11, 12, 14, 16, 19, 20, 21, 23, 25, 27, 28, 31, 32, 32, 32, 32};
    int layers[6] = {0};
    int apr[20] = {0, 0, 0, 0, 0, 2, 4, 2, 2, 2, 2, 3, 1, 2, 2, 2, 3, 1, 1, 1};
    int active[5] = {0};
    int V = 20;
    int L = 5;
    int *d_off, *d_csr, *d_apr, *d_layers, *d_aid, *d_active;
    unsigned int*d_act;

    cudaMalloc(&d_off, 21 * sizeof(int));
    cudaMemcpy(d_off, off, 21*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_apr, 20*sizeof(int));
    cudaMemcpy(d_apr, apr, 20*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_csr, 32*sizeof(int));
    cudaMemcpy(d_csr, csr, 32*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_layers, 6 * sizeof(int));
    // cudaMemcpy(d_layers, layers, 6*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_active, 5 * sizeof(int));
    // cudaMemcpy(d_active, active, 5*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_act, 20*sizeof(unsigned int));
    cudaMemset(d_act, 0, 20*sizeof(unsigned int));
    
    cudaMalloc(&d_aid, 20*sizeof(int));

    cudaMemset(d_aid, 0, 20*sizeof(int));
    cudaMemset(d_layers, 0, 6*sizeof(int));
    cudaMemset(d_active, 0, 5*sizeof(int));

    int blocksize = 1024;
    int nblocks = ceil((float) V/blocksize);

    printf("%d %d\n", blocksize, nblocks);

    Layer0<<<nblocks,blocksize>>>(d_csr, d_off, d_apr, d_layers, V);
    for(int l = 2; l<6; l++)
    {
        Layers<<<nblocks,blocksize>>>(d_csr, d_off, d_apr, d_layers, l);
    }

    for(int l = 0; l< 5; l++)
    {
        Act<<<nblocks,blocksize>>>(d_apr, d_aid, d_act, d_layers, l);
        DeAct<<<nblocks,blocksize>>>(d_apr, d_aid, d_act, d_layers, l);
        AIDs<<<nblocks,blocksize>>>(d_csr, d_off, d_apr, d_aid, d_act, d_layers, l);
        Count<<<nblocks,blocksize>>>(d_act, d_active, d_layers, l);
    }

    cudaMemcpy(layers, d_layers, 6*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(active, d_active, 5*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i =0 ; i< 5; i++)
    {
        printf("%d %d\n", i, active[i]);
    }
}