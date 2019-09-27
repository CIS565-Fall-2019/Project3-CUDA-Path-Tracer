#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        #define blockSize 128

        __global__ void kernNaiveScan(int N, int d, int* read, int* write)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) return;

            int start = pow(float(2), float(d - 1));
            if (index >= start)
            {
                write[index] = read[index - start] + read[index];
            }
            else
            {
                write[index] = read[index];
            }
        }

        __global__ void kernInclusiveToExclusive(int N, int* read, int* write)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) return;

            if (index == 0)
            {
                write[index] = 0;
            }
            else
            {
                write[index] = read[index-1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_read;
            int* dev_write;
            
            cudaMalloc((void**)&dev_read, n * sizeof(int));
            cudaMalloc((void**)&dev_write, n * sizeof(int));

            cudaMemcpy(dev_read, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();
            
            for (int i = 1; i <= ilog2ceil(n); i++)
            {
                kernNaiveScan << <n, blockSize >> > (n, i, dev_read, dev_write);
                std::swap(dev_read, dev_write);
            }
            kernInclusiveToExclusive << <n, blockSize >> > (n, dev_read, dev_write);
   
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_write, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_read);
            cudaFree(dev_write);
        }
    }
}
