#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_launch_parameters.h"


#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

//#define blockSize 128

int* dev_idata;
int* padded_idata;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
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

		__global__ void upSweep(int n, int d, int* A) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			int stride = 1 << (d+1);
			int other_index = 1 << d;
			if ((index) % stride == 0) {
				A[index + stride - 1] += A[index + other_index - 1];
			}
		}

		__global__ void upSweepOptimized(int n, int d, int* A) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			
			
			int other_index = 1 << d; 
			int stride = other_index*2;

			int new_index = stride * index;
			if (new_index >= n) {
				return;
			}
			A[new_index + stride - 1] += A[new_index + other_index - 1];
		}

		__global__ void downSweep(int n, int d, int* A) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			int left_index = 1 << (d);
			int right_index = 1 << (d + 1);
			if (index % right_index == 0) {
				int temp = A[index + left_index - 1];
				A[index + left_index - 1] = A[index + right_index - 1];
				A[index + right_index - 1] += temp;
			}
		}

		__global__ void downSweepOptimized(int n, int d, int* A) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			
			int left_index = 1 << (d);
			int right_index = left_index*2;

			int new_index = right_index * index;
			if (new_index >= n) {
				return;
			}

			int temp = A[new_index + left_index - 1];
			A[new_index + left_index - 1] = A[new_index + right_index - 1];
			A[new_index + right_index - 1] += temp;
			
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, int blockSize) {
			int padded_size = 1 << (ilog2ceil(n));

			cudaMalloc((void**)&padded_idata, padded_size * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc padded_idata failed!");

			cudaMemset(padded_idata, 0, padded_size * sizeof(int));
			cudaMemcpy(padded_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			
			bool caught = false;
			try {
				timer().startGpuTimer();
			}
			catch (const std::exception& e) {
				caught = true;
			}
			

			
			int iterations = ilog2(padded_size);
			dim3 fullBlocksPerGrid((padded_size + blockSize - 1) / blockSize);

			bool optimized = false;
			//Up-Sweep
			if (optimized) {
				int number_of_threads = padded_size;
				for (int d = 0; d < iterations; d++) {
					number_of_threads /= 2;
					dim3 fullBlocksPerGridUpSweep((number_of_threads + blockSize - 1) / blockSize);
					upSweepOptimized << <fullBlocksPerGridUpSweep, blockSize >> > (padded_size, d, padded_idata);
				}
			}else{
				for (int d = 0; d < iterations; d++) {
					dim3 fullBlocksPerGrid((padded_size + blockSize - 1) / blockSize);
					upSweep << <fullBlocksPerGrid, blockSize >> > (padded_size, d, padded_idata);
				}
			}
			
			
			//Down-Sweep
			cudaMemset(padded_idata + (padded_size - 1), 0, sizeof(int));
			
			
			if (optimized) {
				int number_of_threads = 1;
				for (int d = iterations - 1; d >= 0; d--) {
					dim3 fullBlocksPerGridDownSweep((number_of_threads + blockSize - 1) / blockSize);
					downSweepOptimized << <fullBlocksPerGridDownSweep, blockSize >> > (padded_size, d, padded_idata);
					number_of_threads *= 2;
				}
			}
			else {
				for (int d = iterations - 1; d >= 0; d--) {
					dim3 fullBlocksPerGrid((padded_size + blockSize - 1) / blockSize);
					downSweep << <fullBlocksPerGrid, blockSize >> > (padded_size, d, padded_idata);
				}
			}
			if (!caught) {
				timer().endGpuTimer();
			}

			cudaMemcpy(odata, padded_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);

			cudaFree(padded_idata);
        }

		void scan_device(int n, int *odata, const int *idata, int blockSize) {
			int padded_size = 1 << (ilog2ceil(n));

			cudaMalloc((void**)&padded_idata, padded_size * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc padded_idata failed!");

			cudaMemset(padded_idata, 0, padded_size * sizeof(int));
			cudaMemcpy(padded_idata, idata, sizeof(int) * n, cudaMemcpyDeviceToDevice);

			bool caught = false;
			try {
				timer().startGpuTimer();
			}
			catch (const std::exception& e) {
				caught = true;
			}



			int iterations = ilog2(padded_size);
			dim3 fullBlocksPerGrid((padded_size + blockSize - 1) / blockSize);

			bool optimized = false;
			//Up-Sweep
			if (optimized) {
				int number_of_threads = padded_size;
				for (int d = 0; d < iterations; d++) {
					number_of_threads /= 2;
					dim3 fullBlocksPerGridUpSweep((number_of_threads + blockSize - 1) / blockSize);
					upSweepOptimized << <fullBlocksPerGridUpSweep, blockSize >> > (padded_size, d, padded_idata);
				}
			}
			else {
				for (int d = 0; d < iterations; d++) {
					dim3 fullBlocksPerGrid((padded_size + blockSize - 1) / blockSize);
					upSweep << <fullBlocksPerGrid, blockSize >> > (padded_size, d, padded_idata);
				}
			}


			//Down-Sweep
			cudaMemset(padded_idata + (padded_size - 1), 0, sizeof(int));


			if (optimized) {
				int number_of_threads = 1;
				for (int d = iterations - 1; d >= 0; d--) {
					dim3 fullBlocksPerGridDownSweep((number_of_threads + blockSize - 1) / blockSize);
					downSweepOptimized << <fullBlocksPerGridDownSweep, blockSize >> > (padded_size, d, padded_idata);
					number_of_threads *= 2;
				}
			}
			else {
				for (int d = iterations - 1; d >= 0; d--) {
					dim3 fullBlocksPerGrid((padded_size + blockSize - 1) / blockSize);
					downSweep << <fullBlocksPerGrid, blockSize >> > (padded_size, d, padded_idata);
				}
			}
			if (!caught) {
				timer().endGpuTimer();
			}

			cudaMemcpy(odata, padded_idata, sizeof(int) * n, cudaMemcpyDeviceToDevice);

			cudaFree(padded_idata);
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
        int compact(int n, int *odata, const int *idata, int blockSize) {
            

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			int* dev_bools;
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bools failed!");
			int* bools;
			bools = (int*)malloc(n * sizeof(int));
			int* indices;
			indices = (int*)malloc(n * sizeof(int));
			int *dev_indices;
			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bools failed!");
			
			timer().startGpuTimer();
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			
			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);

			
			cudaMemcpy(bools, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);
			
			
			scan(n, indices, bools, blockSize);
			
			int output_length = indices[n - 1] + bools[n-1];

			
			cudaMemcpy(dev_indices, indices, sizeof(int) * n, cudaMemcpyHostToDevice);

			int *dev_odata;
			cudaMalloc((void**)&dev_odata, output_length * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

			cudaMemcpy(odata, dev_odata, sizeof(int) * output_length, cudaMemcpyDeviceToHost);

            timer().endGpuTimer();

			cudaFree(dev_bools);
			cudaFree(dev_indices);
			cudaFree(dev_odata);
            return output_length;
        }
    }
}
