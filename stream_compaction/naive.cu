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
        // TODO: __global__
		__global__ void kernNaiveScan(int n, int offset, int *odata, const int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			
			if (index >= n) {
				return;
			}
			
			if (index >= offset) {
				odata[index] = idata[index] + idata[index-offset];
			}
			else {
				odata[index] = idata[index];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int* dev_data1;
			int* dev_data2;
			cudaMalloc((void**)&dev_data1, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_data1 failed!");

			cudaMalloc((void**)&dev_data2, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_data2 failed!");

			cudaMemcpy(dev_data1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpyHostToDevice failed!");

            timer().startGpuTimer();
            // TODO
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			for (int d = 0; d < ilog2ceil(n); d++) {
				if (d % 2 == 0) {
					kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, 1 << d, dev_data2, dev_data1);
				}
				else {
					kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, 1 << d, dev_data1, dev_data2);
				}
			}

            timer().endGpuTimer();

			if ((ilog2ceil(n) % 2) == 1) {
				cudaMemcpy(odata + 1, dev_data2, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
				checkCUDAErrorFn("cudaMemcpyDeviceToHost) failed!");
			}
			else {
				cudaMemcpy(odata + 1, dev_data1, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
				checkCUDAErrorFn("cudaMemcpyDeviceToHost) failed!");
			}
			
			odata[0] = 0;
			cudaFree(dev_data1);
			cudaFree(dev_data2);
        }
    }
}
