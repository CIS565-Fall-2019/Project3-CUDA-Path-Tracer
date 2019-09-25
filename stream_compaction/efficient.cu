#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "device_launch_parameters.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernelUpSweepStep(int n, int d, int* cdata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k > n)
				return;
			int prev_step_size = 1 << d;
			int cur_step_size = 2 * prev_step_size;
			if (k % cur_step_size == 0)
				cdata[k + cur_step_size - 1] += cdata[k + prev_step_size - 1];
		}

		__global__ void kernelUpSweepStepEfficient(int n, int d, int* cdata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= n)
				return;
			int prev_step_size = 1 << d;
			int cur_step_size = 2 * prev_step_size;
			int new_offset = k * cur_step_size;
			cdata[new_offset + cur_step_size - 1] += cdata[new_offset + prev_step_size - 1];
		}

		__global__ void kernelDownSweepStep(int n, int d, int* cdata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k > n)
				return;
			int left_step = 1 << d;
			int cur_step = 2 * left_step;

			if (k % cur_step == 0) {
				int temp = cdata[k + left_step - 1];
				cdata[k + left_step - 1] = cdata[k + cur_step - 1];
				cdata[k + cur_step - 1] += temp;
			}
		}

		__global__ void kernelDownSweepStepEfficient(int n, int d, int* cdata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= n)
				return;

			int prev_step_size = 1 << d;
			int cur_step_size = 2 * prev_step_size;
			int new_offset = k * cur_step_size;
			
			int temp = cdata[new_offset + prev_step_size - 1];
			cdata[new_offset + prev_step_size - 1] = cdata[new_offset + cur_step_size - 1];
			cdata[new_offset + cur_step_size - 1] += temp;
		}

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

		void printCudaArray(int size, int* data) {
			int *d_data = new int[size];
			cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyDeviceToHost);
			printArray(size, d_data, true);
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scanEfficient(int n, int *odata, const int *idata, int blockSize) {
			// Memory Allocation and Copying
			int power_size = pow(2, ilog2ceil(n));
			int *cdata;
			cudaMalloc((void**)&cdata, power_size * sizeof(int));
			checkCUDAErrorFn("cudaMalloc adata failed!");
			cudaMemset(cdata, 0, power_size * sizeof(int));
			cudaMemcpy(cdata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			bool started_timer = true;
			try {
				timer().startGpuTimer();
			}
			catch (const std::exception& e) {
				started_timer = false;
			}

			int numThreads;
			//Up Sweep
			for (int d = 0; d <= ilog2ceil(power_size) - 1 ; d++) {
				numThreads = pow(2, (ilog2ceil(power_size) - 1 - d));
				dim3 fullBlocks((numThreads + blockSize - 1) / blockSize);
				kernelUpSweepStepEfficient <<<fullBlocks, blockSize>>> (numThreads, d, cdata);
			}

			//Down Sweep
			cudaMemset(cdata + power_size - 1, 0, sizeof(int));
			for (int d = ilog2(power_size) - 1; d >= 0; d--) {
				numThreads = pow(2, (ilog2ceil(power_size) - 1 - d));
				dim3 fullBlocks((numThreads + blockSize - 1) / blockSize);
				kernelDownSweepStepEfficient <<<fullBlocks, blockSize>>> (numThreads, d, cdata);
			}

			if (started_timer)
				timer().endGpuTimer();

			// Copy Back and Free Memory
			cudaMemcpy(odata, cdata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(cdata);
        }

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		 void scanEfficientCUDA(int n, int *odata, const int *idata, int blockSize) {
			 // Memory Allocation and Copying
			 int power_size = pow(2, ilog2ceil(n));
			 int *cdata;
			 cudaMalloc((void**)&cdata, power_size * sizeof(int));
			 checkCUDAErrorFn("cudaMalloc adata failed!");
			 cudaMemset(cdata, 0, power_size * sizeof(int));
			 cudaMemcpy(cdata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

			 bool started_timer = true;
			 try {
				 timer().startGpuTimer();
			 }
			 catch (const std::exception& e) {
				 started_timer = false;
			 }

			 int numThreads;
			 //Up Sweep
			 for (int d = 0; d <= ilog2ceil(power_size) - 1; d++) {
				 numThreads = pow(2, (ilog2ceil(power_size) - 1 - d));
				 dim3 fullBlocks((numThreads + blockSize - 1) / blockSize);
				 kernelUpSweepStepEfficient << <fullBlocks, blockSize >> > (numThreads, d, cdata);
			 }

			 //Down Sweep
			 cudaMemset(cdata + power_size - 1, 0, sizeof(int));
			 for (int d = ilog2(power_size) - 1; d >= 0; d--) {
				 numThreads = pow(2, (ilog2ceil(power_size) - 1 - d));
				 dim3 fullBlocks((numThreads + blockSize - 1) / blockSize);
				 kernelDownSweepStepEfficient << <fullBlocks, blockSize >> > (numThreads, d, cdata);
			 }

			 if (started_timer)
				 timer().endGpuTimer();

			 // Copy Back and Free Memory
			 cudaMemcpy(odata, cdata, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			 cudaFree(cdata);
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata, int blockSize) {
			// Memory Allocation and Copying
			int power_size = pow(2, ilog2ceil(n));
			int *cdata;
			cudaMalloc((void**)&cdata, power_size * sizeof(int));
			checkCUDAErrorFn("cudaMalloc adata failed!");
			cudaMemset(cdata, 0, power_size * sizeof(int));
			cudaMemcpy(cdata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			bool started_timer = true;
			try {
				timer().startGpuTimer();
			}
			catch (const std::exception& e) {
				started_timer = false;
			}
			dim3 fullBlocksPerGrid((power_size + blockSize - 1) / blockSize);

			//Up Sweep
			for (int d = 0; d < ilog2ceil(power_size); d++) {
				kernelUpSweepStep << <fullBlocksPerGrid, blockSize >> > (power_size, d, cdata);
			}

			//Down Sweep
			cudaMemset(cdata + power_size - 1, 0, sizeof(int));

			for (int d = ilog2(power_size) - 1; d >= 0; d--) {
				kernelDownSweepStep << <fullBlocksPerGrid, blockSize >> > (power_size, d, cdata);
			}
			if (started_timer)
				timer().endGpuTimer();

			// Copy Back and Free Memory
			cudaMemcpy(odata, cdata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(cdata);
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
        int compact(int n, int *odata, const int *idata, bool efficient, int blockSize) {
			// Memory Allocation and Copying
			int *bools = new int[n];
			int *indices = new int[n];
			int *dev_bools;
			int *dev_indices;
			int *dev_idata;
			int *dev_odata;
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_bools failed!");
			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_indices failed!");
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_odata failed!");
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			timer().startGpuTimer();
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);
			cudaMemcpy(bools, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);
			if(efficient)
				scanEfficient(n, indices, bools, blockSize);
			else
				scan(n, indices, bools, blockSize);
			cudaMemcpy(dev_indices, indices, sizeof(int) * n, cudaMemcpyHostToDevice);
			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);			
			timer().endGpuTimer();

			// Copy Back and Free Memory
			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_bools);
			cudaFree(dev_indices);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
            return indices[n - 1] + bools[n - 1];;
        }
    }
}
