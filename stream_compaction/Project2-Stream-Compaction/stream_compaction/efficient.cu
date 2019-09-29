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

		int *dev_arrayA;
		int *dev_arrayB;

		int *dev_bools;
		int *dev_boolScans;

		int *dev_idata;
		int *dev_odata;
		
		int * dev_indices;

		int *dev_lastElements;
		int *dev_lastElements2;

		void printArray(int n, const int *a, bool abridged = false) {
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

		__global__ void kernEffScanUpSweep(int N, int pow2d, int pow2d1, int* arrA) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= N) return;

			if ((k % pow2d1) == 0 && (k + pow2d1 - 1)<N && (k + pow2d - 1)<N ){
				arrA[k + pow2d1 - 1] += arrA[k + pow2d - 1];
			}
		}

		__global__ void kernEffScanDownSweep(int N, int pow2d, int pow2d1, int* arrA) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= N) return;
			
			int tmp = 0;
			
			if ((k % pow2d1) == 0 && (k + pow2d1 - 1) < N && (k + pow2d - 1) < N) {
				tmp = arrA[k + pow2d -1];
				arrA[k + pow2d - 1] = arrA[k + pow2d1 - 1];
				arrA[k + pow2d1 - 1] += tmp;
			}
		}

		__global__ void kernInitZero(int N, int* array) {
			
			int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
			
			if (tid < N) {
				array[tid] = 0;
			}
		}

		__global__ void kernScanShared(int n, int * g_odata, int * g_idata) {
		
			extern __shared__ int temp[];  // allocated on invocation
			
			int thid = threadIdx.x;
			int tid_read = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (tid_read >= n) return;

			int offset = 1;

			temp[2 * thid] = g_idata[2 * tid_read]; // load input into shared memory
			temp[2 * thid + 1] = g_idata[2 * tid_read + 1];

			// build sum in place up the tree
			for (int d = 2*blockDim.x >> 1; d > 0; d >>= 1)
			{
				__syncthreads();

				if (thid < d)
				{
					int ai = offset * (2 * thid + 1) - 1;
					int bi = offset * (2 * thid + 2) - 1;

					temp[bi] += temp[ai];
				}
				offset *= 2;
			}

			if (thid == 0) { temp[2 * blockDim.x - 1] = 0; } // clear the last element

			for (int d = 1; d < 2 * blockDim.x; d *= 2) // traverse down tree & build scan
			{
				offset >>= 1;
				__syncthreads();

				if (thid < d)
				{
					int ai = offset * (2 * thid + 1) - 1;
					int bi = offset * (2 * thid + 2) - 1;
					
					int t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;

				}
			}

			__syncthreads();

			g_odata[2 * tid_read] = temp[2 * thid]; // write results to device memory
			g_odata[2 * tid_read + 1] = temp[2 * thid + 1];
		}

		__global__ void kernGetLastElement(int n, int* s_data, int * g_odata, int * g_idata) {
			int thid = threadIdx.x;

			int tid_global = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (tid_global >= n) return;

			if (thid == blockDim.x - 1) {
				s_data[blockIdx.x] = g_odata[tid_global] +g_idata[tid_global];
			}
		}

		__global__ void kernUpdateScan(int n, int* s_data, int * g_odata, int * g_idata) {
			int thid = threadIdx.x;
			int tid_global = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (tid_global >= n) return;

			g_odata[tid_global]  += s_data[blockIdx.x];

		}

        /*
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
		/*
        void scan(int n, int *odata, const int *idata) {

            // TODO
			int n_new = n;

			//check for non-2powerN
			if (1 << ilog2ceil(n) != n)
				n_new = (1 << ilog2ceil(n));

			int fullBlocksPerGrid((n_new + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_arrayA, n_new * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_arrayA failed!");

			//Initialize to Zero
			kernInitZero <<<fullBlocksPerGrid, blockSize >>> (n_new, dev_arrayA);
			checkCUDAErrorFn("kernInitZero failed!");

			// Fill dev_arrayA with idata
			cudaMemcpy(dev_arrayA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpyToSymbol from idata to dev_arrayA failed!");

			bool tmp = true;
			try {
				timer().startGpuTimer();
				//printf("IN WEScan timer started!\n");
			}
			catch (const std::runtime_error& e) {
				tmp = false;
			}

			// Upstream
			int pow2d1 = 0;
			int pow2d = 0;
			for (int d = 0; d <= ilog2ceil(n_new)-1; d++) {
				pow2d = 1 << (d);
				pow2d1 = 1 << (d+1);
				kernEffScanUpSweep << <fullBlocksPerGrid, blockSize >> > (n_new, pow2d, pow2d1, dev_arrayA);
				checkCUDAErrorFn("kernEffScanUpSweep failed!");		
			}

			// Downstream
			int *zero = new int[1];
			zero[0] = 0;
			cudaMemcpy(dev_arrayA + n_new-1, zero, 1*sizeof(int), cudaMemcpyHostToDevice);

			for (int d = ilog2ceil(n_new)-1; d >= 0; d--) {
				pow2d = 1 << (d);
				pow2d1 = 1 << (d + 1);
				kernEffScanDownSweep << <fullBlocksPerGrid, blockSize >> > (n_new, pow2d, pow2d1, dev_arrayA);
				checkCUDAErrorFn("kernGenerateRandomPosArray failed!");
			}

			if (tmp == true) { 
				timer().endGpuTimer();
				//printf("IN WEScan timer ended!\n");
			}

			// Copy back to cpu
			cudaMemcpy(odata, dev_arrayA, n*sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_arrayA to odata failed!");
			
			//printf("BBT Scan Final Computed : \n");
			//printArray(n, odata, true);

			cudaFree(dev_arrayA);
			return;
        }
		*/


		void oldScan(int n_new, int *odata, int *idata) {

			int fullBlocksPerGrid((n_new + blockSize - 1) / blockSize);

			// Upstream
			int pow2d1 = 0;
			int pow2d = 0;
			for (int d = 0; d <= ilog2ceil(n_new) - 1; d++) {
				pow2d = 1 << (d);
				pow2d1 = 1 << (d + 1);
				kernEffScanUpSweep << <fullBlocksPerGrid, blockSize >> > (n_new, pow2d, pow2d1, idata);
				checkCUDAErrorFn("kernEffScanUpSweep failed!");
			}

			// Downstream
			int *zero = new int[1];
			zero[0] = 0;
			cudaMemcpy(idata + n_new - 1, zero, 1 * sizeof(int), cudaMemcpyHostToDevice);

			for (int d = ilog2ceil(n_new) - 1; d >= 0; d--) {
				pow2d = 1 << (d);
				pow2d1 = 1 << (d + 1);
				kernEffScanDownSweep << <fullBlocksPerGrid, blockSize >> > (n_new, pow2d, pow2d1, idata);
				checkCUDAErrorFn("kernGenerateRandomPosArray failed!");
			}

			// Copy back to out
			cudaMemcpy(odata, idata, n_new * sizeof(int), cudaMemcpyDeviceToDevice);
			checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_arrayB to odata failed!");
			return;
		}


		void scan(int n, int *odata, const int *idata) {

			// TODO
			int n_new = n;
			//int *tmp_print = new int[n];

			//check for non-2powerN
			if (1 << ilog2ceil(n) != n) 
				n_new = (1 << ilog2ceil(n));
			
			int fullBlocksPerGrid((n_new + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_arrayA, n_new * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_arrayA failed!");

			//Initialize to Zero
			kernInitZero << <fullBlocksPerGrid, blockSize >> > (n_new, dev_arrayA);
			checkCUDAErrorFn("kernInitZero failed!");

			// Fill dev_arrayA with idata
			cudaMemcpy(dev_arrayA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpyToSymbol from idata to dev_arrayA failed!");

			// More arrays
			cudaMalloc((void**)&dev_odata, n_new * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_arrayA failed!");

			cudaMalloc((void**)&dev_lastElements, n_new * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_arrayA failed!");

			cudaMalloc((void**)&dev_lastElements2, n_new * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_arrayA failed!");

			bool tmp = true;
			try {
				timer().startGpuTimer();
				//printf("IN WEScan timer started!\n");
			}
			catch (const std::runtime_error& e) {
				tmp = false;
			}

			//printf("\n==========================STARTED WES================================\n");
			//printf("Pre Scan Array \n");
			//printArray(n, idata, true);

			//fullBlocksPerGrid = 4;

			kernScanShared <<< fullBlocksPerGrid, blockSize / 2, (2*blockSize + blockSize/8) * sizeof(int) >> > (n_new, dev_odata, dev_arrayA);
			//cudaMemcpy(tmp_print, dev_odata, n_new * sizeof(int), cudaMemcpyDeviceToHost);
			//checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_odata to tmp_print failed!");
			//printf("kernScanShared results per %d blocks\n", fullBlocksPerGrid);
			//printArray(n_new, tmp_print, true);

			kernGetLastElement << < fullBlocksPerGrid, blockSize, blockSize * sizeof(int) >> > (n_new, dev_lastElements, dev_odata, dev_arrayA);
			//cudaMemcpy(tmp_print, dev_lastElements, fullBlocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);
			//checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_odata to odata failed!");
			//printf("kernGetLastElement results\n");
			//printArray(fullBlocksPerGrid, tmp_print, true);

			oldScan(fullBlocksPerGrid, dev_lastElements2, dev_lastElements);
			//kernScanShared << < 1, blockSize / 2, blockSize * sizeof(int) >> > (n_new, dev_lastElements2, dev_lastElements);
			//cudaMemcpy(tmp_print, dev_lastElements2, fullBlocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);
			//checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_odata to odata failed!");
			//printf("scan on  kernGetLastElement\n");
			//printArray(fullBlocksPerGrid, tmp_print, true);

			kernUpdateScan << < fullBlocksPerGrid, blockSize >> > (n_new, dev_lastElements2, dev_odata, dev_arrayA);
			//cudaMemcpy(tmp_print, dev_odata, n_new * sizeof(int), cudaMemcpyDeviceToHost);
			//checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_odata to odata failed!");
			//printf("FINAL Scan results\n");
			//printArray(n_new, tmp_print, true);
			//printf("\n==========================FINISHED WES================================\n");


			if (tmp == true) {
				timer().endGpuTimer();
				//printf("IN WEScan timer ended!\n");
			}

			// Copy back
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_arrayA to odata failed!");

			//printf("BBT Scan Final Computed : \n");
			//printArray(n, odata, true);
			cudaFree(dev_arrayA);
			cudaFree(dev_odata);
			cudaFree(dev_lastElements);
			cudaFree(dev_lastElements2);

			return;
		}



		void compact_scan(int n, int *dev_odata, int *dev_idata) {

			// TODO
			int n_new = n;
			//int *tmp_print = new int[n];

			//check for non-2powerN
			if (1 << ilog2ceil(n) != n) {
				n_new = (1 << ilog2ceil(n));
			}
			
			int fullBlocksPerGrid((n_new + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_arrayA, n_new * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_arrayA failed!");

			//Initialize to Zero
			kernInitZero <<<fullBlocksPerGrid, blockSize >> > (n_new, dev_arrayA);
			checkCUDAErrorFn("kernInitZero failed!");

			// Fill dev_arrayA with idata
			cudaMemcpy(dev_arrayA, dev_idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
			checkCUDAErrorFn("cudaMemcpyToSymbol from idata to dev_arrayA failed!");
			
			// More arrays
			cudaMalloc((void**)&dev_lastElements, n_new * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_arrayA failed!");

			cudaMalloc((void**)&dev_lastElements2, n_new * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_arrayA failed!");

			bool tmp = true;
			try {
				timer().startGpuTimer();
				//printf("IN WEScan timer started!\n");
			}
			catch (const std::runtime_error& e) {
				tmp = false;
			}

			kernScanShared << < fullBlocksPerGrid, blockSize / 2, (2 * blockSize + blockSize / 8) * sizeof(int) >> > (n_new, dev_odata, dev_arrayA);
			//cudaMemcpy(tmp_print, dev_odata, n_new * sizeof(int), cudaMemcpyDeviceToHost);

			kernGetLastElement << < fullBlocksPerGrid, blockSize, blockSize * sizeof(int) >> > (n_new, dev_lastElements, dev_odata, dev_arrayA);
			//cudaMemcpy(tmp_print, dev_lastElements, fullBlocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

			oldScan(fullBlocksPerGrid, dev_lastElements2, dev_lastElements);
			//kernScanShared << < 1, blockSize / 2, blockSize * sizeof(int) >> > (n_new, dev_lastElements2, dev_lastElements);

			kernUpdateScan << < fullBlocksPerGrid, blockSize >> > (n_new, dev_lastElements2, dev_odata, dev_arrayA);
			//cudaMemcpy(tmp_print, dev_odata, n_new * sizeof(int), cudaMemcpyDeviceToHost);

			if (tmp == true) {
				timer().endGpuTimer();
				//printf("IN WEScan timer ended!\n");
			}

			// Copy back
			//cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			//checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_arrayA to odata failed!");

			//printf("BBT Scan Final Computed : \n");
			//printArray(n, odata, true);
			cudaFree(dev_arrayA);
			cudaFree(dev_lastElements);
			cudaFree(dev_lastElements2);

			return;
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
           
            // TODO
			int * indices = new int[n];
			int * bools = new int[n];
			int fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_bools failed!");

			cudaMalloc((void**)&dev_idata, n*sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_arrayA failed!");

			cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpyToSymbol from idata to dev_arrayA failed!");

			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_indices failed!");

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_indices failed!");


			timer().startGpuTimer();
			
			//Compute bools
			Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize >>>(n, dev_bools, dev_idata);
			checkCUDAErrorFn("kernMapToBoolean failed!");

			//compute scans
			compact_scan(n, dev_indices, dev_bools);

			//scatter
			Common::kernScatter<<<fullBlocksPerGrid, blockSize >>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
			checkCUDAErrorFn("kernScatter failed!");
			
			timer().endGpuTimer();

			// Copy back to cpu
			cudaMemcpy(odata, dev_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_odata to odata failed!");


			int *lastEl = new int[1];
			cudaMemcpy(lastEl, dev_indices+n-1, 1*sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_odata to odata failed!");

			//printf("GPU Compaction : \n");
			//printArray(n, odata, true);

			cudaFree(dev_bools);
			cudaFree(dev_idata);
			cudaFree(dev_indices);
			cudaFree(dev_odata);

			if (idata[n - 1] != 0) {
				return lastEl[0] + 1;
			}
			else {
				return lastEl[0];
			}
        }		
	}
}
