#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		__global__ void kernScan(int n, int p, int *odata, int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			if (index >= p) {
				odata[index] = idata[index - p] + idata[index];
			}
			else {
				odata[index] = idata[index];
			}

		}

		__global__ void kernRightShift(int n, int *odata, int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			if (index == 0) {
				odata[index] = 0;
			}
			odata[index + 1] = idata[index];
			
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			//set up variable, allocate space on gpu, and copy over data
			int *dev_idata;
			int *dev_odata;

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");

			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("Memcpy idata failed!");

			dim3 threadsPerBlock(blockSize);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			timer().startGpuTimer();

			//inclusive scanning
			for (int d = 0; d < ilog2ceil(n); d++) {
				int p = 1 << d;
				kernScan<<<fullBlocksPerGrid, threadsPerBlock >>>(n, p, dev_odata, dev_idata);
				checkCUDAError("kernel kernScan failed!");

				std::swap(dev_idata, dev_odata);
			}

			//right shift to convert to exclusive scan
			kernRightShift << <fullBlocksPerGrid, threadsPerBlock >> > (n, dev_odata, dev_idata);
			checkCUDAError("kernel   kernRightShift failed!");

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAError("Memcpy odata failed!");

			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }
    }
}
