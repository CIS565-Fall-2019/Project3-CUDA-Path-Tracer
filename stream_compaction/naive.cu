#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }
		//initial
		/*__global__ void kernInitialArray(int N, int * arr) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;//threadindex
			if (index >= N) {
				return;
			}
			arr[index] = -1;
		}*/

        // TODO: __global__
		__global__ void kernNaiveScan(int n, int d, int *odata, int* idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			//do the naive scan
			int start = powf(2.0, 1.0*(d - 1));//2^(d-1)
			if (index >= start) {
				odata[index] = idata[index - start] + idata[index];
			} else {
				odata[index] = idata[index];
			}
		}

		__global__ void kernInclu2Exclu(int n, int *inclu, int* exclu) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			if (index == 0) {
				exclu[index] = 0;
			} else {
				exclu[index] = inclu[index - 1];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {         
			int *dev_idata;
			int *dev_odata;

			//malloc memory
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			//checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			//checkCUDAError("cudaMalloc dev_odata failed!");

			//mempy
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);//host to device
			//checkCUDAError("cudaMemcpy dev_idata failed!");
			cudaMemcpy(dev_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);//host to device
			//checkCUDAError("cudaMemcpy dev_odata failed!");

			int d = ilog2ceil(n);//log2 n
			int* temp;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			//start time 
			timer().startGpuTimer();
			for (int i = 1; i <= d; i++) {
				kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, i, dev_odata, dev_idata);
				temp = dev_idata;
				dev_idata = dev_odata;
				dev_odata = temp;
			}
			//from inclusive to exclusive 
			kernInclu2Exclu << <fullBlocksPerGrid, blockSize >> > (n, dev_idata, dev_odata);
            // TODO
            timer().endGpuTimer();
			//end gpu time

			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);//get the result
			//checkCUDAError("get odata failed!");

			//free
			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }
    }
}
