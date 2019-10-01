#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernUpSweep(int n, int offset, int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			idata[(index+1)*offset*2 - 1] += idata[offset*(2*index + 1) - 1];
		}

		__global__ void kernDownSweep(int n, int offset, int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			int t = idata[offset*(2 * index + 1) - 1];
			idata[offset*(2 * index + 1) - 1] = idata[(index + 1)*offset * 2 - 1];
			idata[(index + 1)*offset * 2 - 1] += t;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int tempSize = pow(2, ilog2ceil(n));
			int* dev_idata;
			cudaMalloc((void**)&dev_idata, tempSize * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int),cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpyHostToDevice failed!");

            timer().startGpuTimer();
            // TODO
			int level = ilog2ceil(n);
			dim3 fullBlocksPerGrid;

			for (int d = 0; d < level; d++) {
				int threads = 1 << (level - 1 - d);
				fullBlocksPerGrid = dim3((threads + blockSize - 1) / blockSize);
				kernUpSweep << <fullBlocksPerGrid, blockSize >> > (threads, 1 << d, dev_idata);
			}

			cudaMemset(dev_idata + tempSize - 1, 0, sizeof(int));

			for (int d = level - 1; d >= 0; d--) {
				int threads = 1 << (level - 1 - d);
				fullBlocksPerGrid = dim3((threads + blockSize - 1) / blockSize);
				kernDownSweep << <fullBlocksPerGrid, blockSize >> > (threads, 1 << d, dev_idata);
			}

			timer().endGpuTimer();
			cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyDeviceToHost failed!");
			cudaFree(dev_idata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

			int tempSize = pow(2, ilog2ceil(n));
			int *dev_indices, *dev_bools, *dev_odata, *dev_idata;

			cudaMalloc((void**)&dev_indices, tempSize * sizeof(int));
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMalloc((void**)&dev_idata, n * sizeof(int));

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
			dim3 fullBlocksPerGrid = (n + blockSize - 1) / blockSize;
			Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);
			
			dim3 fullBlocksPerGrid2;
			int level = ilog2ceil(tempSize);
			cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
			int threads;

			for (int d = 0; d < level; d++) {
				threads = 1 << (level - 1 - d);
				fullBlocksPerGrid2 = dim3((threads + blockSize - 1) / blockSize);
				kernUpSweep << <fullBlocksPerGrid2, blockSize >> > (threads, 1 << d, dev_indices);
			}

			cudaMemset(dev_indices + tempSize - 1, 0, sizeof(int));

			for (int d = level - 1; d >= 0; d--) {
				threads = 1 << (level - 1 - d);
				fullBlocksPerGrid2 = dim3((threads + blockSize - 1) / blockSize);
				kernDownSweep << <fullBlocksPerGrid2, blockSize >> > (threads, 1 << d, dev_indices);
			}
			Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();
			int *bools = (int*)malloc(n * sizeof(int));
			int *indices = (int*)malloc(n * sizeof(int));
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(indices, dev_indices, n * sizeof(int), cudaMemcpyDeviceToHost);

			int remain = bools[n - 1] ? indices[n - 1] + 1 : indices[n - 1];

			cudaFree(dev_bools);
			cudaFree(dev_indices);
			cudaFree(dev_odata);
			cudaFree(dev_idata);
			free(bools);
			free(indices);

            return remain;
        }
    }
}
