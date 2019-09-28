#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

# define blockSize 1024

namespace StreamCompaction {
	namespace Efficient {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		int *dev_arr1;
		int *dev_arr2;
		int *dev_bools;
		int *dev_indices;
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

		__global__ void kernUpSweep(int n, int valPower2D, int *data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n)
				return;

			if (index % (2 * valPower2D) == 0 && (index + (2 * valPower2D) - 1 < n) && (index + valPower2D - 1 < n)) {
				data[index + (2 * valPower2D) - 1] += data[index + valPower2D - 1];
			}
		}

		__global__ void kernZeroPadding(int n, int N, int *data) {
			
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >=  N- n)
				return;
			data[index + n] = 0;

		}
		__global__ void kernLastElement(int n, int *data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index == n - 1)
				data[index] = 0;
			else
				return;
		}
		__global__ void kernDownSweep(int n, int valPower2D, int *data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n)
				return;

			if ((index % (2 * valPower2D) == 0) && (index + (2 * valPower2D) - 1 < n) && (index + valPower2D - 1 < n)) {
				int temp = data[index + valPower2D - 1];
				data[index + valPower2D - 1] = data[index + (2 * valPower2D) - 1];
				data[index + (2 * valPower2D) - 1] += temp;
			}
		}

		__global__ void treeScanSharedMem(int n, int *inData, int *outData) {

			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= n/2)
				return;
			
			extern __shared__ int sharedMem[];

			sharedMem[2 * threadIdx.x] = inData[2 * idx];
			sharedMem[2 * threadIdx.x + 1 ] = inData[2 * idx + 1];
			int newIndex;
			int oldIndex;
			int multiplier;

			// Upsweep
			for (int d = 1; d < n; d = d << 1) {
				//printf("For thread id: %d, the value of d is: %d\n", idx, d);
				multiplier = n / (2 * d);
				if (idx < multiplier) {
					newIndex = d *( 2 * threadIdx.x + 1) - 1;
					oldIndex = d *( 2 * threadIdx.x + 2) - 1;
					sharedMem[oldIndex] += sharedMem[newIndex];
				}
				__syncthreads();
			}
			
			
			// Downsweep
			if (idx == n/2 - 1)
				sharedMem[blockDim.x * (threadIdx.x +1) - 1] = 0;
			
			int temp;
			int d = n / 2;
			for (int i = 1 ; i < n ; i = i << 1 ) {
				
				if (idx < i) {
					newIndex = d *( 2 * threadIdx.x + 1) - 1;
					oldIndex = d *( 2 * threadIdx.x + 2) - 1;
					temp = sharedMem[newIndex];
					sharedMem[newIndex] = sharedMem[oldIndex];
					sharedMem[oldIndex] += temp;
				}
				d = d >> 1;
				__syncthreads();
				
			}
			__syncthreads();
			outData[2 * idx] = sharedMem[2 * threadIdx.x];
			outData[2 * idx + 1] = sharedMem[2 * threadIdx.x + 1];

		}
		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {

			int diff = (1 << ilog2ceil(n)) - n;
			int N = n + diff;

			int *dev_arr2;

			cudaMalloc((void**)&dev_arr1, N * sizeof(int));
			checkCUDAErrorFn("Malloc idata into arr1 failed");

			cudaMalloc((void**)&dev_arr2, N * sizeof(int));
			checkCUDAErrorFn("Malloc odata into arr2 failed");

			cudaMemcpy(dev_arr1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("Copying idata to arr1 failed");

			dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

			bool stopTimer = false;
			try {
				timer().startGpuTimer();
			}
			catch (const std::runtime_error& exception) {
				stopTimer = true;
			}

			dim3 fullBlocksPerGrid2((N - n + blockSize - 1) / blockSize);
			if (diff) {
				kernZeroPadding << <fullBlocksPerGrid2, blockSize >> > (n, N, dev_arr1);
			}
			cudaDeviceSynchronize();
			/*
			for (int d = 0; d <= ilog2ceil(n) - 1; d++) {
				int valPower2D = 1 << d;
				kernUpSweep << <fullBlocksPerGrid, blockSize >> > (N, valPower2D, dev_arr1);
				checkCUDAErrorFn("Kernel Up Sweep Failed");
			}

			kernLastElement << <fullBlocksPerGrid, blockSize >> > (N, dev_arr1);

			for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
				int valPower2D = 1 << d;
				kernDownSweep << <fullBlocksPerGrid, blockSize >> > (N, valPower2D, dev_arr1);
				checkCUDAErrorFn("Kernel Down Sweep Failed");
			}
			*/

			dim3 fullBlocksPerGrid3((N/2 + blockSize - 2) / blockSize);
			treeScanSharedMem << <fullBlocksPerGrid3, blockSize ,N*sizeof(int)>> > (N, dev_arr1, dev_arr2);
			checkCUDAErrorFn("Tree scan Shared mem");
			cudaDeviceSynchronize();
			if (!stopTimer)
				timer().endGpuTimer();

			//printf("Hello 2\n");
			cudaMemcpy(odata, dev_arr2, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("Copying back to Host failed");

			//printf("Upsweep output is \n");
			//printArray(N, odata, true);

			cudaFree(dev_arr1);
			cudaFree(dev_arr2);


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

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_arr2, n * sizeof(int));
			checkCUDAErrorFn("Malloc idata into arr2 failed");

			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAErrorFn("Malloc idata into arr3 failed");

			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			checkCUDAErrorFn("Malloc idata into indices failed");

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorFn("Malloc idata into odata failed");

			cudaMemcpy(dev_arr2, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("Copying idata to arr2 failed");

			timer().startGpuTimer();

			Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_arr2);
			checkCUDAErrorFn("Kernel Map indicator failed");

			int *indices = new int[n];
			int *bools = new int[n];

			cudaMemcpy(bools, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("Copying bools to host failed");

			scan(n, indices, bools);

			cudaMemcpy(dev_indices, indices, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("Copying indices to device failed");

			Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_arr2, dev_bools, dev_indices);
			checkCUDAErrorFn("Kernel Scatter failed");

			timer().endGpuTimer();

			int length = indices[n - 1];

			if (idata[n - 1])
				length += 1;

			//printf("Length is %d \n", length);
			cudaMemcpy(odata, dev_odata, sizeof(int) * length, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("Copying back to the host failed");

			cudaFree(dev_arr2);
			cudaFree(dev_bools);
			cudaFree(dev_indices);
			cudaFree(dev_odata);
			return length;
		}
	}
}