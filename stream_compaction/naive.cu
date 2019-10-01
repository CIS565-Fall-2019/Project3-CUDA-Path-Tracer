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
        
		__global__ void kernShiftForZero(const int n, int *odata, const int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n) {
				index == 0 ? odata[index] = 0 : odata[index] = idata[index - 1];
			}
		}

		__global__ void kernScanNaive(int n, int d, int *odata, const int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n) {
				
				int exp = (int) powf(2.f, float(d) - 1.f);
				if (index >= exp) {
					odata[index] = (idata[index - exp] + idata[index]);
				} else {
					odata[index] = idata[index];
				}
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Initializing arrays
			int* dev_tempIData;
			int* dev_tempOData;

			cudaMalloc((void**)&dev_tempIData, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc tempIData failed!", "Naive.cu", 44);

			cudaMalloc((void**)&dev_tempOData, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc tempOData failed!", "Naive.cu", 47);

			// Copying the input array to the device
			cudaMemcpy(dev_tempIData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy to input-copy failed!", "Naive.cu", 51);

			timer().startGpuTimer();

			// Using GPU Gems' scan function
			for (int d = 1; d <= ilog2ceil(n); d++) {
				kernScanNaive << < n, blockSize >> > (n, d, dev_tempOData, dev_tempIData);
				checkCUDAErrorFn("kernScanNaive failed!", "Naive.cu", 58);

				std::swap(dev_tempIData, dev_tempOData);
			}

			// Make it an exclusive scan
			kernShiftForZero << < n, blockSize >> > (n, dev_tempOData, dev_tempIData);
			checkCUDAErrorFn("kernShiftForZero failed!", "Naive.cu", 65);

			timer().endGpuTimer();

			// Copying the result back to the output array
			cudaMemcpy(odata, dev_tempOData, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy to output failed!", "Naive.cu", 71);

            cudaFree(dev_tempIData);
			cudaFree(dev_tempOData);
        }

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 Inclusive to assist with Radix sort
		 */
		void scanInclusive(int n, int *odata, const int *idata) {
			// Initializing arrays
			int* dev_tempIData;
			int* dev_tempOData;

			cudaMalloc((void**)&dev_tempIData, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc tempIData failed!", "Naive.cu", 44);

			cudaMalloc((void**)&dev_tempOData, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc tempOData failed!", "Naive.cu", 47);

			// Copying the input array to the device
			cudaMemcpy(dev_tempIData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy to input-copy failed!", "Naive.cu", 51);

			// Using GPU Gems' scan function
			for (int d = 1; d <= ilog2ceil(n); d++) {
				kernScanNaive << < n, blockSize >> > (n, d, dev_tempOData, dev_tempIData);
				checkCUDAErrorFn("kernScanNaive failed!", "Naive.cu", 58);

				std::swap(dev_tempIData, dev_tempOData);
			}


			// Copying the result back to the output array
			cudaMemcpy(odata, dev_tempOData, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy to output failed!", "Naive.cu", 71);

			cudaFree(dev_tempIData);
			cudaFree(dev_tempOData);
		}
    }
}
