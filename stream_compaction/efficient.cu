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
			if (index >= N - n)
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
			int thid = threadIdx.x;
			if (idx >= n/2 )
				return;

			//printf("Index value id: %d\n", idx);
			
			extern __shared__ int sharedMem[];

			sharedMem[2 * thid] = inData[2 * idx];
			sharedMem[2 * thid + 1] = inData[2 * idx + 1];
			int ia;
			int ib;
			int multiplier;
			int offset = 1;
			//int numBlocks = n / (2 * blockDim.x);

			// Upsweep
			for (int d = 2*blockDim.x >> 1 ; d > 0; d = d >> 1) {
				//printf("For thread id: %d, the value of d is: %d\n", idx, d);
				__syncthreads();
				if (thid < d) {
					ia = offset * (2 * thid + 1) - 1;
					ib = offset * (2 * thid + 2) - 1;
					//printf("Access %d, %d, %d, %d \n", thid, idx, ia, ib);
					sharedMem[ib] += sharedMem[ia];
				}

				offset *= 2;
			}


			// Downsweep
			if (thid == 0)
				sharedMem[2*blockDim.x -1] = 0;

			int temp;
			//int d = n / 2;
			for (int i = 1; i < 2*blockDim.x; i = i << 1) {
				offset >>= 1;
				if (thid < i) {
					ia = offset * (2 * thid + 1) - 1;
					ib = offset * (2 * thid + 2) - 1;
					temp = sharedMem[ia];
					sharedMem[ia] = sharedMem[ib];
					sharedMem[ib] += temp;
				}
				__syncthreads();

			}

			__syncthreads();
			outData[2 * idx] = sharedMem[2 * thid];
			outData[2 * idx + 1] = sharedMem[2 * thid + 1];


		}

		__global__ void kernLastElementBlockSum(int N, int* indata,int* odata,int* buffer) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			int thid = threadIdx.x;

			if (idx >= N)
				return;

			//printf("The value is: %d\n", idx);

			if (thid == blockDim.x - 1) {
				buffer[blockIdx.x] = odata[idx];
				buffer[blockIdx.x] += indata[idx];
			}
			

			//printf("The buffer data is %d \n", buffer[thid]);
		}

		__global__ void kernAddEachBlockElement(int N, int* buffer_data, int* odata) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= N)
				return;

			odata[idx] += buffer_data[blockIdx.x];

		}
		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void oldscan(int n, int *dev_arr2, int *dev_arr1) {
			int diff = (1 << ilog2ceil(n)) - n;
			int N = n + diff;

			//int *dev_arr2;
			//int *dev_buffer;
			//int *dev_buffer_2;

			//cudaMalloc((void**)&dev_arr1, N * sizeof(int));
			//checkCUDAErrorFn("Malloc idata into arr1 failed");

			//cudaMalloc((void**)&dev_arr2, N * sizeof(int));
			//checkCUDAErrorFn("Malloc odata into arr2 failed");

			//cudaMemcpy(dev_arr1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			//checkCUDAErrorFn("Copying idata to arr1 failed");

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

			if (!stopTimer)
				timer().endGpuTimer();

			//printf("Hello 2\n");
			cudaMemcpy(dev_arr2, dev_arr1, sizeof(int) * N, cudaMemcpyDeviceToDevice);
			checkCUDAErrorFn("Copying back to Host failed");

			//printf("\n Output is \n");
			//printArray(N, odata, true);

			//cudaFree(dev_arr1);
			
		}
		void scan(int n, int *odata, const int *idata) {

			int diff = (1 << ilog2ceil(n)) - n;
			int N = n + diff;

			int *dev_arr2;
			int *dev_buffer;
			int *dev_buffer_2;

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

			//printf("Input Array \n");
			//printArray(N, idata, true);
			int blkSize = blockSize;
			int numBlocks = (N + blockSize -1 )/ blockSize;
			//dim3 fullBlocksPerGrid3((N / 2 + blkSize - 1) / blkSize);
			treeScanSharedMem << <numBlocks, blkSize/2, blkSize*sizeof(int) >> > (N, dev_arr1, dev_arr2);
			checkCUDAErrorFn("Tree scan Shared mem ghg");
			cudaDeviceSynchronize();

			/*
			int *temp0 = new int[N];

			cudaMemcpy(temp0, dev_arr2, sizeof(int) * N, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("Copying back to Host failed jfkgf");

			printf("\n Each block scan output is \n");
			printArray(N, temp0, true);
			*/

			cudaMalloc((void**)&dev_buffer, numBlocks * sizeof(int));
			checkCUDAErrorFn("Malloc odata into arr2 failed");

			kernLastElementBlockSum << <numBlocks, blkSize >> > (N, dev_arr1, dev_arr2, dev_buffer);
			checkCUDAErrorFn("Kernel last element failed");

			/*
			int *temp = new int[numBlocks];
			
			cudaMemcpy(temp, dev_buffer, sizeof(int) * numBlocks, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("Copying back to Host failed jfkgf");

			printf("\n Buffer output is \n");
			printArray(numBlocks, temp, true);

	        */
			cudaMalloc((void**)&dev_buffer_2, numBlocks * sizeof(int));
			checkCUDAErrorFn("Malloc odata into arr2 failed");

			oldscan(numBlocks,dev_buffer_2,dev_buffer);

			/*
			int *temp2 = new int[numBlocks];

			cudaMemcpy(temp2, dev_buffer_2, sizeof(int) * numBlocks, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("Copying back to Host failed jfkgf");

			printf("\n Scan of scan output is \n");
			printArray(numBlocks, temp2, true);
			*/

			kernAddEachBlockElement << < numBlocks, blkSize >> > (N,dev_buffer_2,dev_arr2);
			checkCUDAErrorFn("Adding Element to Each Block failed");

			/*
			int *temp2 = new int[numBlocks];

			cudaMemcpy(temp2, dev_buffer, sizeof(int) * numBlocks, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("Copying back to Host failed jfkgf");

			printf("\n Buffer output is \n");
			printArray(numBlocks, temp, true);
			*/

			/*
			kernLastElementBlockSum << <fullBlocksPerGrid3,blockSize >> > (N,dev_arr1,dev_arr2,dev_buffer);
			checkCUDAErrorFn("Kernel last element failed");

			do {
				
				treeScanSharedMem << <fullBlocksPerGrid3, blockSize, N * sizeof(int) >> > (N, dev_buffer, dev_buffer_out);
				checkCUDAErrorFn("Tree scan Shared mem failed");

				kernAddEachBlockElement << <fullBlocksPerGrid3, blockSize >> (dev_buffer_out,outdata);
				checkCUDAErrorFn("Kernel addition back failed");

			} while ();
			*/

			if (!stopTimer)
				timer().endGpuTimer();

			//printf("Hello 2\n");
			cudaMemcpy(odata, dev_arr2, sizeof(int) * N, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("Copying back to Host failed");

			//printf("\n Output is \n");
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