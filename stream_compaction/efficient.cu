#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "common.h"
#include "device_launch_parameters.h"
#include "efficient.h"

namespace StreamCompaction {
	namespace Shared {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		__global__ void addKernel(int power_size, int* cdata, int* second_level) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= power_size)
				return;

			cdata[k] += second_level[blockIdx.x];
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

		/*
		 * Scan using global memory 
		 */
		void scan(int n, int *cdata, int blockSize) {
			// Memory Allocation and Copying
			int power_size = 1 << ilog2ceil(n);
			int numThreads;
			//Up Sweep
			for (int d = 0; d <= ilog2ceil(power_size) - 1; d++) {
				numThreads = 1 << (ilog2ceil(power_size) - 1 - d);
				dim3 fullBlocks((numThreads + blockSize - 1) / blockSize);
				kernelUpSweepStepEfficient << <fullBlocks, blockSize >> > (numThreads, d, cdata);
			}

			//Down Sweep
			cudaMemset(cdata + power_size - 1, 0, sizeof(int));
			for (int d = ilog2(power_size) - 1; d >= 0; d--) {
				numThreads = 1 << (ilog2ceil(power_size) - 1 - d);
				dim3 fullBlocks((numThreads + blockSize - 1) / blockSize);
				kernelDownSweepStepEfficient << <fullBlocks, blockSize >> > (numThreads, d, cdata);
			}
		}

		/*
		 * Kernel that scans the array using the shared memory
		 */
		__global__ void scanKernelShared(int power_size, int* cdata, int* second_level, const int blockSize) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= power_size)
				return;
			// Copy Data to Shared Memory
			__shared__ int s[1024];
			s[threadIdx.x] = cdata[k];
			__syncthreads();

			//Up Sweep
			for (int d = 0; d <= ilog2ceil(blockSize) - 1; d++) {
				__syncthreads();
				int prev_step_size = 1 << d;
				int cur_step_size = 2 * prev_step_size;
				if (threadIdx.x % cur_step_size == 0) {
					s[threadIdx.x + cur_step_size - 1] += s[threadIdx.x + prev_step_size - 1];
				}
			}

			// Write the sum of all elements in this block in the second level array
			__syncthreads();
			if (threadIdx.x == 0) {
				second_level[blockIdx.x] = s[blockSize - 1];
				s[blockSize - 1] = 0;
			}

			//Down Sweep
			for (int d = ilog2(blockSize) - 1; d >= 0; d--) {
				__syncthreads();
				int left_step = 1 << d;
				int cur_step = 2 * left_step;

				if (threadIdx.x % cur_step == 0) {
					int temp = s[threadIdx.x + left_step - 1];
					s[threadIdx.x + left_step - 1] = s[threadIdx.x + cur_step - 1];
					s[threadIdx.x + cur_step - 1] += temp;
				}
			}
			cdata[k] = s[threadIdx.x];
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scanEfficient(int n, int *odata, const int *idata, int blockSize) {
			// Memory Allocation and Copying
			int power_size = pow(2, ilog2ceil(n));
			int num_blocks = (power_size + blockSize - 1) / blockSize;
			int *cdata, *second_level;
			cudaMalloc((void**)&cdata, power_size * sizeof(int));
			checkCUDAErrorFn("cudaMalloc adata failed!");
			cudaMalloc((void**)&second_level, num_blocks * sizeof(int));
			checkCUDAErrorFn("cudaMalloc second_level failed!");
			cudaMemset(cdata, 0, power_size * sizeof(int));
			cudaMemcpy(cdata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			bool started_timer = true;
			try {
				timer().startGpuTimer();
			}
			catch (const std::exception& e) {
				started_timer = false;
			}

			dim3 fullBlocks(num_blocks);
			scanKernelShared << <fullBlocks, blockSize >> > (power_size, cdata, second_level, blockSize);

			dim3 level2Blocks((num_blocks + blockSize - 1) / blockSize);
			scan(num_blocks, second_level, blockSize);

			addKernel << <fullBlocks, blockSize >> > (power_size, cdata, second_level);

			if (started_timer)
				timer().endGpuTimer();

			// Copy Back and Free Memory
			cudaMemcpy(odata, cdata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(cdata);
			cudaFree(second_level);
		}

		/**
		 * Performs stream compaction on idata, storing the result into odata.
		 * All zeroes are discarded.
		 *
		 * @param n      The number of elements in idata.
		 * @param dev_idata  The array of elements to compact inplace
		 * @returns      The number of elements remaining after compaction.
		 */
		int compactCUDA(int n, int *dev_idata) {
			int blockSize = 1024;
			// Memory Allocation and Copying
			int *bools = new int[n];
			int *indices = new int[n];
			int *dev_odata;
			int *dev_bools;
			int *dev_indices;
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_bools failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_indices failed!");

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);
			cudaMemcpy(bools, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);
			scanEfficient(n, indices, bools, blockSize);
			cudaMemcpy(dev_indices, indices, sizeof(int) * n, cudaMemcpyHostToDevice);

			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
			cudaMemcpy(dev_idata, dev_odata, sizeof(int) * n, cudaMemcpyHostToDevice);

			// Copy Back and Free Memory
			cudaFree(dev_bools);
			cudaFree(dev_indices);
			cudaFree(dev_odata);
			return indices[n - 1] + bools[n - 1];;
		}
	}
}
