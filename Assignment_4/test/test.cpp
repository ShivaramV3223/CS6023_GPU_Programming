#include<stdio.h>
using namespace std;
# define N 2
# define R 4
# define max_P 3

int main(void)
{
    int center[2] = {0, 1};
    int fac[2] = {0, 2};
    int fac_ids[5] = {0, 1, 0, 1, 2};
    int cap[5] = {1, 2, 1, 2, 2};

    int req_ids[4] = {0, 1, 2, 3};
    int req_cen[4] = {1, 1, 1, 0};
    int req_fac[4] = {0, 0, 1, 1};
    int req_start[4] = {21, 23, 27, 12};
    int req_slots[4] = {3, 2, 4, 2};

    int fac_req[N * max_P * R] = {-1};
    int fac_reqs_c[N * max_P] = {0};
    int tot_reqs[N] = {0};
    int succ_reqs[N] = {0};
    int reqs[R] = {-1};
    int tmp;
    
    // Preprocess kernel
    for(int i=0; i< 4; i++)
    {  
        reqs[i] = fac[req_cen[i]] + req_fac[i];
        printf("%d ", reqs[i]);
    }
    printf("\n");

    for(int i=0; i< 4; i++)
    {
        for(int j =i + 1; j <3; j++)
        {
            if(reqs[i] == reqs[j])
            {
                
            }
        }
    }
}