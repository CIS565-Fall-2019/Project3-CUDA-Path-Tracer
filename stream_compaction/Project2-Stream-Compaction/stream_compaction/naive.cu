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

		int *dev_arrayA; 
		int *dev_arrayB;

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
		__global__ void kernPrefixSumScanArray(int N, int pow2d1, int* arrA, int*arrB) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= N) return;
			
			if (k >= pow2d1) {
				arrB[k] = arrA[k - (pow2d1)] + arrA[k];			
			}
		}

		__global__ void kernExclusiveShiftArray(int N,  int* arrA, int*arrB) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			
			if (k >= N) return;

			if (k == 0) {
				arrA[0] = 0;
			}
			else {
				arrA[k] = arrB[k-1];
			}
		}

		/**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
			int fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			
			cudaMalloc((void**)&dev_arrayA, n*sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_arrayA failed!");

			cudaMalloc((void**)&dev_arrayB, n*sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_arrayB failed!");
			
			// Fill dev_arrayA with idata
			cudaMemcpy(dev_arrayA, idata, n*sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpyToSymbol from idata to dev_arrayA failed!");
			
			// Fill dev_arrayB with idata
			cudaMemcpy(dev_arrayB, idata, n*sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpyToSymbol from idata to dev_arrayB failed!");

			timer().startGpuTimer();

			// Call Scan Kernel
			int pow2d1 = 0;

			for (int d = 1; d <= ilog2ceil(n); d++) {
				pow2d1 = 1 << (d - 1);
				kernPrefixSumScanArray<<<fullBlocksPerGrid, blockSize>>>(n, pow2d1, dev_arrayA, dev_arrayB);
				checkCUDAErrorFn("kernGenerateRandomPosArray failed!");
				
				//Copy
				cudaMemcpy(dev_arrayA, dev_arrayB, n*sizeof(int), cudaMemcpyDeviceToDevice);
			}

			kernExclusiveShiftArray <<<fullBlocksPerGrid, blockSize >>> (n, dev_arrayA, dev_arrayB);
			checkCUDAErrorFn("kernExclusiveShiftArray failed!");

			timer().endGpuTimer();

			// Fill dev_arrayA with idata
			cudaMemcpy(odata, dev_arrayA, n*sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_arrayA to odata failed!");

			//printf("Final Computed after shifting: \n");
			//printArray(n, odata, true);
			//printf("Computed: \n");

			cudaFree(dev_arrayA);
			cudaFree(dev_arrayB);
		}
    }
}
