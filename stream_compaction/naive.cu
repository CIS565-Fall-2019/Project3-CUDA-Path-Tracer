#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "common.h"
#include "naive.h"

#define blockSize 512


namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		int *dev_idata;
		int *dev_odata;

		void printArray(int n, int *a, bool abridged = false) {
			printf("    [ ");
			for (int i = 0; i < n; i++) {
				if (abridged && i + 2 == 15 && n > 16) {
					i = n - 2;
					printf("... ");
				}
				printf("%3d ", a[i]);
			}
			printf("]\n");
		}

        // TODO: __global__

		__global__ void scan_GPU(int N, int *Dev_idata, int *Dev_odata, int d) {

			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index >= N) {
				return;
			}
				


			if (index >= (1 << (d - 1))) {
				Dev_odata[index] = Dev_idata[index - (1 << (d - 1))] + Dev_idata[index];
			}
			else {
				Dev_odata[index] = Dev_idata[index];
			}
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			bool timer_started = false;
			
			//printArray(n, idata);
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMemcpy(dev_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);

			try {
				timer().startGpuTimer();
			}
			catch (const std::exception& e) {
				timer_started = true;
			}

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);


			for (int d = 1; d <= ilog2ceil(n); d++) {
				scan_GPU << <fullBlocksPerGrid, blockSize >> > (n, dev_idata, dev_odata, d);
				cudaMemcpy(dev_idata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			}

			if (timer_started == false) {
				timer().endGpuTimer();
			}
			cudaMemcpy(odata+1, dev_odata, sizeof(int) * (n-1), cudaMemcpyDeviceToHost);
			odata[0] = 0;
			//printArray(n, odata);
			cudaFree(dev_odata);
			cudaFree(dev_idata);
        }
    }
}
