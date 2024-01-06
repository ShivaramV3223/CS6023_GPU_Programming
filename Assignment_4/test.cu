#include <iostream>
#include <cuda.h>
#include <stdio.h>

using namespace std;

__global__ void prefix_sum(int * arr, int N)
{
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  for(int off = 1; off < N ; off *= 2)
  {
    if(idx >= off)
    {
      int val = idx % (2 * off);

      if(val >= off)
      {
        arr[idx] += arr[idx - val + off -1];
      }
    }
  }
}

int main(void)
{
    int facs[5] = {2, 3, 5, 6 , 9};
    int *dfacs;

    cudaMalloc(&dfacs, 5 * sizeof(int));
    cudaMemcpy(dfacs, facs, 5 * sizeof(int), cudaMemcpyHostToDevice);

    prefix_sum<<<1,5>>>(dfacs, 5);

    cudaMemcpy(facs, dfacs, 5 * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i =0; i< 5; i++)
    {
        printf("%d\n", facs[i]);
    }
}