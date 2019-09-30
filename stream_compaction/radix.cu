#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "efficient.h"
#include <thrust/scan.h>
#include "radix.h"

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
  
		__global__ void kernFullAddress(int n, int totalFalses, int *indices, int *bools){
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			indices[index] = bools[index] ? indices[index] : index - indices[index] + totalFalses;
		}

		__global__ void kernBitMapToBoolean(int n, int bit, int *bools, const int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			bools[index] = idata[index] & bit ? 0 : 1;
		}

		__global__ void kernFullScatter(int n, int *odata, const int *idata, const int *indices) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			odata[indices[index]] = idata[index];
			
		}
		
		int findMax(int n, const int *idata) {
			int mx = idata[0];
			for (int i = 1; i < n; i++) {
				if (idata[i] > mx) {
					mx = idata[i];
				}
			}
			return mx;
		}

        void sort(int n, int *odata, const int *idata) {
			//int mx = findMax(n, idata);
			int kbit = ilog2ceil(n);
			int npad = 1 << kbit;

			int *dev_idata, *dev_odata, *dev_bools, *dev_indices;
			cudaMalloc((void**)&dev_idata, sizeof(int) * npad);
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, sizeof(int) * npad);
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&dev_bools, sizeof(int) * npad);
			checkCUDAError("cudaMalloc dev_bools failed!");
			cudaMalloc((void**)&dev_indices, sizeof(int) * npad);
			checkCUDAError("cudaMalloc dev_indices failed!");

			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("Memcpy idata failed!");
			
			dim3 threadsPerBlock(blockSize);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			
            timer().startGpuTimer();
            for(int i = 0; i <= kbit; i++){
				kernBitMapToBoolean<< <fullBlocksPerGrid, threadsPerBlock >> > (n, (1 << i), dev_bools, dev_idata);
				checkCUDAError("Memcpy kernMapToBoolean failed!");

				cudaMemcpy(dev_indices, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);
				checkCUDAError("Memcpy dev_bools to dev_indices failed!");
				StreamCompaction::Efficient::workEfficientScan(npad, dev_indices, threadsPerBlock, fullBlocksPerGrid);
				checkCUDAError("Efficient scan failed!");

				int totalFalses, lastBool;
				cudaMemcpy(&totalFalses, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
				checkCUDAError("Memcpy totalFalses failed!");
				cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
				checkCUDAError("Memcpy lastBool failed!");

				totalFalses += lastBool;
				kernFullAddress << <fullBlocksPerGrid, threadsPerBlock >> > (n, totalFalses, dev_indices, dev_bools);
				checkCUDAError("kernFullAddress failed!");

				kernFullScatter << <fullBlocksPerGrid, threadsPerBlock >> > (n, dev_odata, dev_idata, dev_indices);
				checkCUDAError("kernScatter failed!");

				cudaMemcpy(dev_idata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToDevice);
				checkCUDAError("Memcpy dev_odata to dev_idata failed!");
			}

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);


			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_bools);
			cudaFree(dev_indices);
        }
    }
}
