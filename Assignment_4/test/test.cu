#include <iostream>
#include <cuda.h>
#include <stdio.h>

# define max_P 30
using namespace std;

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
            printf("req %d is processed by fac %d\n", i, idx);
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


int main(void)
{
    int N = 2;
    int center[2] = {0, 1};
    int fac[2] = {2, 3};
    int fac_ids[5] = {0, 1, 0, 1, 2};
    int cap[5] = {1, 2, 1, 2, 2};

    int req_ids[4] = {0, 1, 2, 3};
    int req_cen[4] = {1, 1, 1, 0};
    int req_fac[4] = {0, 0, 1, 1};
    int req_start[4] = {21, 23, 27, 12};
    int req_slots[4] = {3, 2, 4, 2};

    int fac_[5] = {0};
    int fac_req[5 * 4] = {0};
    int tot_req[2] = {0};
    int succ_req[2] = {0};

    int *d_center, *d_fac, *d_fac_ids, *d_cap;
    int *d_req_ids, *d_req_cen, *d_req_fac, *d_req_start, *d_req_slots;
    int *d_fac_, *d_fac_req;
    int *d_tot_req, *d_succ_req;

    cudaMalloc(&d_center, 2 * sizeof(int));
    cudaMalloc(&d_fac, 2 * sizeof(int));
    cudaMalloc(&d_fac_ids, 5 * sizeof(int));
    cudaMalloc(&d_cap, 5 * sizeof(int));

    cudaMalloc(&d_req_ids, 4 * sizeof(int));
    cudaMalloc(&d_req_cen, 4 * sizeof(int));
    cudaMalloc(&d_req_fac, 4 * sizeof(int));
    cudaMalloc(&d_req_start, 4 * sizeof(int));
    cudaMalloc(&d_req_slots, 4 * sizeof(int));

    cudaMalloc(&d_fac_, 5* sizeof(int));
    cudaMalloc(&d_fac_req, 20* sizeof(int));

    cudaMalloc(&d_tot_req, 2 * sizeof(int));
    cudaMalloc(&d_succ_req, 2 * sizeof(int));


    cudaMemcpy(d_center, center, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fac, fac, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fac_ids, fac_ids, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cap, cap, 5 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_req_ids, req_ids, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_cen, req_cen, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_fac, req_fac, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_start, req_start, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_slots, req_slots, 4 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_tot_req, tot_req, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_succ_req, succ_req, 2 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_fac_, fac_, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fac_req, fac_req, 20 * sizeof(int), cudaMemcpyHostToDevice);

    prefix_sum<<<1,2>>>(d_fac, 2);
    cudaMemcpy(fac, d_fac, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i < 2; i++)
    {
        cout << fac[i] << endl;
    }
    int F = fac[N - 1];
    printf("F = %d\n", F);
    Process<<<1,F>>>(d_fac, d_cap, d_req_cen, d_req_fac, d_req_start, d_req_slots, d_tot_req, d_succ_req, 4, F);
    cudaMemcpy(tot_req, d_tot_req, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(succ_req, d_succ_req, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i<2; i++)
    {
        cout << tot_req[i] << " " << succ_req[i] << endl;
    }
    cudaDeviceSynchronize();

    
}