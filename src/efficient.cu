#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
//#include "thrust.h"
#include <thrust/scan.h>
#include <thrust/device_vector.h>

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg)

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		void printxxx(int n, const int *a) {
			for (int i = 0; i < n; i++) {
				printf("%d ", a[i]);
			}
			printf("\n\n\n");
		}

		__global__ void resetZeros(int n, int *a) {
			int index = (blockDim.x*blockIdx.x) + threadIdx.x;
			if (index >= n) return;
			a[index] = 0;
		}



		__global__ void kernscanBlock(int n, int *odata, int* out_last, const int *idata) {
			
			extern __shared__ int temp[];

			int idx = threadIdx.x;
			int tid = (blockDim.x*blockIdx.x) + threadIdx.x;
			int numPerBlock = 2 * blockDim.x;

			if (tid >= n) return;

			// copy the data this idx boi has to work with to shared memory
			temp[2*idx] = idata[2*tid];
			temp[2*idx + 1] = idata[2*tid + 1];

			int offset = 1;
			for (int d = numPerBlock>> 1; d > 0; d >>=1) {
				__syncthreads();

				if (idx < d) {

					int k1 = offset * (2 * idx + 1) - 1;
					int k2 = offset * (2 * idx + 2) - 1;
					temp[k2] += temp[k1];
				}

				offset = 2 * offset;
			}

			if (idx == 0) { temp[numPerBlock - 1] = 0; }

			for (int d = 1; d < numPerBlock; d *= 2) {
				offset >>= 1;
				__syncthreads();
				if (idx < d) {

					int k1 = offset * (2 * idx + 1) - 1;
					int k2 = offset * (2 * idx + 2) - 1;

					int tmp = temp[k1];
					temp[k1] = temp[k2];
					temp[k2] += tmp;
				}
			}

			__syncthreads();
			
			odata[2 * tid] = temp[2 * idx]; // has to updated with block number
			odata[2 * tid + 1] = temp[2 * idx + 1];

			if (idx == 0) {
				int last = numPerBlock * blockIdx.x + numPerBlock - 1;
				out_last[blockIdx.x] = temp[numPerBlock - 1] + idata[last];
			}
		}

		__global__ void copyLastElements(int n, int blockSize, int *odata, const int *idata) {
			int tid = (blockDim.x*blockIdx.x) + threadIdx.x;
			if (tid >= n) return;

			odata[tid] = idata[tid*blockSize + blockSize - 1];
		}

		__global__ void addLastElement(int n, int blockSize, int *odata, const int *scanSum, const int *idata) {
			int tid = (blockDim.x*blockIdx.x) + threadIdx.x;
			if (tid >= n) return;

			odata[tid] = scanSum[tid] + idata[tid*blockSize + blockSize - 1];
			//odata[tid] = scanSum[tid];
		}

		__global__ void addScanMain(int n, int *odata, const int *scanSum, const int *scanSumBlock) {
			int tid = (blockDim.x*blockIdx.x) + threadIdx.x;
			if (tid >= n) return;

			odata[tid] = scanSumBlock[tid] + scanSum[blockIdx.x];
		}


		void scanShared(int n, int *odata, const int *idata) {
			bool exception = false;

			int *dev_idata, *dev_scanSumBlock, *dev_addLastElements, *dev_scanSum, *dev_odata;

			int d_max = ilog2ceil(n);

			int twoPowN = 1 << d_max;
			if (n != twoPowN) {

				int diff = twoPowN - n;

				cudaMalloc((void **)&dev_idata, (n + diff) * sizeof(int));
				checkCUDAErrorWithLine("cudaMalloc dev_odata1 failed!");

				int threadsPerBlock = 1024;
				int blocksToLaunch = (n + diff + threadsPerBlock - 1) / threadsPerBlock;
				resetZeros << <blocksToLaunch, threadsPerBlock >> > (n + diff, dev_idata);

				cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
				n = n + diff;
			}
			else {
				cudaMalloc((void **)&dev_idata, n * sizeof(int));
				checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

				cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			}

			int blockSize = 1024;
			int numBlocks = (n + blockSize - 1) / blockSize;
			int numElements = numBlocks;

			cudaMalloc((void **)&dev_scanSumBlock, n * sizeof(int));
			cudaMalloc((void **)&dev_addLastElements, numElements * sizeof(int));
			cudaMalloc((void **)&dev_scanSum, numElements * sizeof(int));
			cudaMalloc((void **)&dev_odata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc failed!");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			//thrust::device_ptr<int> dev_idataItr(dev_addLastElements);
			//thrust::device_ptr<int> dev_odataItr(dev_scanSum);

			try {
				timer().startGpuTimer();
			}
			catch (const std::runtime_error& ex) {
				exception = true;
			}

			kernscanBlock << <numBlocks, blockSize/2, (blockSize) * sizeof(int) >> > (n, dev_scanSumBlock, dev_addLastElements, dev_idata);

			scanCompact(numElements, dev_scanSum, dev_addLastElements);
			//thrust::exclusive_scan(dev_idataItr, dev_idataItr + numElements, dev_odataItr);

			addScanMain<<<numBlocks, blockSize >>>(n, dev_odata, dev_scanSum, dev_scanSumBlock);

			if (!exception)
				timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_idata);
			cudaFree(dev_scanSum);
			cudaFree(dev_scanSumBlock);
			cudaFree(dev_addLastElements);
			cudaFree(dev_odata);
		}

		void scanSharedGPU(int n, int *dev_odata, const int *idata) {
			bool exception = false;

			int *dev_idata, *dev_scanSumBlock, *dev_addLastElements, *dev_scanSum;

			int d_max = ilog2ceil(n);

			int twoPowN = 1 << d_max;
			if (n != twoPowN) {

				int diff = twoPowN - n;

				cudaMalloc((void **)&dev_idata, (n + diff) * sizeof(int));
				checkCUDAErrorWithLine("cudaMalloc dev_odata1 failed!");

				int threadsPerBlock = 1024;
				int blocksToLaunch = (n + diff + threadsPerBlock - 1) / threadsPerBlock;
				resetZeros << <blocksToLaunch, threadsPerBlock >> > (n + diff, dev_idata);

				cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
				n = n + diff;
			}
			else {
				cudaMalloc((void **)&dev_idata, n * sizeof(int));
				checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

				cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
			}

			int blockSize = 512;
			int numBlocks = (n + blockSize - 1) / blockSize;
			int numElements = numBlocks;

			cudaMalloc((void **)&dev_scanSumBlock, n * sizeof(int));
			cudaMalloc((void **)&dev_addLastElements, numElements * sizeof(int));
			cudaMalloc((void **)&dev_scanSum, numElements * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc failed!");

			//thrust::device_ptr<int> dev_idataItr(dev_addLastElements);
			//thrust::device_ptr<int> dev_odataItr(dev_scanSum);

			try {
				timer().startGpuTimer();
			}
			catch (const std::runtime_error& ex) {
				exception = true;
			}

			kernscanBlock << <numBlocks, blockSize / 2, (blockSize) * sizeof(int) >> > (n, dev_scanSumBlock, dev_addLastElements, dev_idata);

			//int *a = new int[n];
			//cudaMemcpy(a, dev_scanSumBlock, n * sizeof(int), cudaMemcpyDeviceToHost);
			//printxxx(n, a);

			scanCompact(numElements, dev_scanSum, dev_addLastElements);
			//thrust::exclusive_scan(dev_idataItr, dev_idataItr + numElements, dev_odataItr);

			addScanMain << <numBlocks, blockSize >> > (n, dev_odata, dev_scanSum, dev_scanSumBlock);

			if (!exception)
				timer().endGpuTimer();

			cudaFree(dev_idata);
			cudaFree(dev_scanSum);
			cudaFree(dev_scanSumBlock);
			cudaFree(dev_addLastElements);
		}

		__global__ void upSweep(int n, int d, int *idata) {
			int index = (blockDim.x*blockIdx.x) + threadIdx.x;

			int twoPowd1 = 1 << (d + 1);
			int twoPowd = 1 << d;


			if ((index % twoPowd1 != twoPowd1-1) || index >= n) return;

			int k = index - twoPowd1 + 1;
			idata[index] += idata[k + twoPowd - 1];
		}

		__global__ void downSweep(int n, int d, int *idata) {
			int index = (blockDim.x*blockIdx.x) + threadIdx.x;

			int twoPowd1 = 1 << (d + 1);
			int twoPowd = 1 << d;


			if ((index % twoPowd1 != twoPowd1 - 1) || index >= n) return;

			int k = index - twoPowd1 + 1;
			int t = idata[k + twoPowd - 1];
			idata[k + twoPowd - 1] = idata[index];
			idata[index] += t;
		}


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			bool exception = false;

			int *dev_idata;

			int numThreads = 128;
			int numBlocks = (n + numThreads - 1) / numThreads;

			int d_max = ilog2ceil(n);

			int twoPowN = 1 << d_max;
			if (n != twoPowN) {

				int diff = twoPowN - n;

				cudaMalloc((void **)&dev_idata, (n + diff) * sizeof(int));
				checkCUDAErrorWithLine("cudaMalloc dev_odata1 failed!");

				resetZeros << <numBlocks, numThreads >> > (n + diff, dev_idata);

				cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
				n = n + diff;
			} else {
				cudaMalloc((void **)&dev_idata, n * sizeof(int));
				checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

				cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			}

			try {
				timer().startGpuTimer();
			}
			catch (const std::runtime_error& ex) {
				exception = true;
			}


			for (int d = 0; d < d_max; d++) {
				upSweep<<<numBlocks, numThreads>>>(n, d, dev_idata);
			}

			// reset last element to zero
			//int* zero = new int[1];
			//zero[0] = 0;
			//cudaMemcpy(dev_idata + n - 1, zero, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemset(dev_idata + n - 1, 0, sizeof(int));

			
			for(int d = d_max-1; d >= 0; d--) {
				downSweep << <numBlocks, numThreads >> > (n, d, dev_idata);
			}


			if (!exception)
				timer().endGpuTimer();


			cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);


			
			cudaFree(dev_idata);


        }

		void scanCompact(int n, int *odata, const int *idata) {
			bool exception = false;

			int *dev_idata;

			int numThreads = 128;
			int numBlocks = (n + numThreads - 1) / numThreads;

			int d_max = ilog2ceil(n);

			int twoPowN = 1 << d_max;
			if (n != twoPowN) {

				int diff = twoPowN - n;

				cudaMalloc((void **)&dev_idata, (n + diff) * sizeof(int));
				checkCUDAErrorWithLine("cudaMalloc dev_odata1 failed!");

				resetZeros << <numBlocks, numThreads >> > (n + diff, dev_idata);

				cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
				n = n + diff;
			}
			else {
				cudaMalloc((void **)&dev_idata, n * sizeof(int));
				checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

				cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
			}

			try {
				timer().startGpuTimer();
			}
			catch (const std::runtime_error& ex) {
				exception = true;
			}


			for (int d = 0; d < d_max; d++) {
				upSweep << <numBlocks, numThreads >> > (n, d, dev_idata);
			}

			// reset last element to zero
			//int* zero = new int[1];
			//zero[0] = 0;
			//cudaMemcpy(dev_idata + n - 1, zero, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemset(dev_idata + n - 1, 0, sizeof(int));


			for (int d = d_max - 1; d >= 0; d--) {
				downSweep << <numBlocks, numThreads >> > (n, d, dev_idata);
			}

			if (!exception)
				timer().endGpuTimer();

			cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToDevice);



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
            

			int numThreads = 128;
			int numBlocks = (n + numThreads - 1) / numThreads;

			int *dev_checkZeros, *dev_sumIndices, *dev_odata, *dev_idata;

			cudaMalloc((void **) &dev_checkZeros, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_checkZeros failed!");
			cudaMalloc((void **) &dev_sumIndices, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_sumIndices failed!");
			cudaMalloc((void **)&dev_odata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");
			cudaMalloc((void **)&dev_idata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			timer().startGpuTimer();

			StreamCompaction::Common::kernMapToBoolean<<<numBlocks, numThreads>>>(n, dev_checkZeros, dev_idata);
			
			int *checkZeros = new int[n];
			cudaMemcpy(checkZeros, dev_checkZeros, n * sizeof(int), cudaMemcpyDeviceToHost);


			int *sumIndices = new int[n];
			scan(n, sumIndices, checkZeros);

			cudaMemcpy(dev_sumIndices, sumIndices , n * sizeof(int), cudaMemcpyHostToDevice);

			StreamCompaction::Common::kernScatter<<<numBlocks, numThreads>>>(n, dev_odata, dev_idata, dev_checkZeros, dev_sumIndices);

			
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			

			int count = checkZeros[n - 1] == 0 ? sumIndices[n - 1] : sumIndices[n - 1] + 1;

			//delete[] checkZeros;
			//delete[] sumIndices;

			//printf("hey\n");

			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_checkZeros);
			cudaFree(dev_sumIndices);

           
            return count;
        }


		int compactShared(int n, int *dev_idata) {


			int numThreads = 128;
			int numBlocks = (n + numThreads - 1) / numThreads;

			int *dev_checkZeros, *dev_sumIndices, *dev_odata;

			cudaMalloc((void **)&dev_checkZeros, n * sizeof(int));
			cudaMalloc((void **)&dev_sumIndices, n * sizeof(int));
			cudaMalloc((void **)&dev_odata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

			//cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			timer().startGpuTimer();

			StreamCompaction::Common::kernMapToBoolean << <numBlocks, numThreads >> > (n, dev_checkZeros, dev_idata);

			//int *a = new int[n];
			//cudaMemcpy(a, dev_checkZeros, n * sizeof(int), cudaMemcpyDeviceToHost);
			//printxxx(n, a);

			scanSharedGPU(n, dev_sumIndices, dev_checkZeros);
			//thrust::device_ptr<int> i1(dev_checkZeros);
			//thrust::device_ptr<int> o1(dev_sumIndices);
			//thrust::exclusive_scan(i1, i1 + n, o1);

			//int *a = new int[n];
			//cudaMemcpy(a, dev_sumIndices, n * sizeof(int), cudaMemcpyDeviceToHost);
			//printxxx(n, a);

			StreamCompaction::Common::kernScatter << <numBlocks, numThreads >> > (n, dev_odata, dev_idata, dev_checkZeros, dev_sumIndices);

			timer().endGpuTimer();

			cudaMemcpy(dev_idata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);

			int *sumIndices = new int[1];
			cudaMemcpy(sumIndices, dev_sumIndices + n-1, 1 * sizeof(int), cudaMemcpyDeviceToHost);
			int *checkZeros = new int[1];
			cudaMemcpy(checkZeros, dev_checkZeros + n-1, 1 * sizeof(int), cudaMemcpyDeviceToHost);

			int count = checkZeros[0] == 0 ? sumIndices[0] : sumIndices[0] + 1;

			delete[] checkZeros;
			delete[] sumIndices;

			cudaFree(dev_odata);
			cudaFree(dev_checkZeros);
			cudaFree(dev_sumIndices);

			return count;
		}


		//int compact(int n, int *odata, const int *idata) {


		//	int numThreads = 128;
		//	int numBlocks = (n + numThreads - 1) / numThreads;

		//	int *dev_checkZeros, *dev_sumIndices, *dev_odata, *dev_idata;

		//	cudaMalloc((void **)&dev_checkZeros, n * sizeof(int));
		//	checkCUDAErrorWithLine("cudaMalloc dev_checkZeros failed!");
		//	cudaMalloc((void **)&dev_sumIndices, n * sizeof(int));
		//	checkCUDAErrorWithLine("cudaMalloc dev_sumIndices failed!");
		//	cudaMalloc((void **)&dev_odata, n * sizeof(int));
		//	checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");
		//	cudaMalloc((void **)&dev_idata, n * sizeof(int));
		//	checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

		//	cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

		//	timer().startGpuTimer();

		//	StreamCompaction::Common::kernMapToBoolean << <numBlocks, numThreads >> > (n, dev_checkZeros, dev_idata);

		//	//int *checkZeros = new int[n];
		//	//cudaMemcpy(checkZeros, dev_checkZeros, n * sizeof(int), cudaMemcpyDeviceToHost);


		//	//int *sumIndices = new int[n];
		//	scanCompact(n, dev_sumIndices, dev_checkZeros);

		//	//cudaMemcpy(dev_sumIndices, sumIndices, n * sizeof(int), cudaMemcpyHostToDevice);

		//	StreamCompaction::Common::kernScatter << <numBlocks, numThreads >> > (n, dev_odata, dev_idata, dev_checkZeros, dev_sumIndices);


		//	timer().endGpuTimer();

		//	cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

		//	int *sumIndices = new int[n];
		//	int *checkZeros = new int[n];
		//	cudaMemcpy(checkZeros, dev_checkZeros, n * sizeof(int), cudaMemcpyDeviceToHost);
		//	cudaMemcpy(sumIndices, dev_sumIndices, n * sizeof(int), cudaMemcpyDeviceToHost);
		//	int count = checkZeros[n - 1] == 0 ? sumIndices[n - 1] : sumIndices[n - 1] + 1;

		//	//delete[] checkZeros;
		//	//delete[] sumIndices;

		//	//printf("hey\n");

		//	cudaFree(dev_idata);
		//	cudaFree(dev_odata);
		//	cudaFree(dev_checkZeros);
		//	cudaFree(dev_sumIndices);

		//	delete[] sumIndices;
		//	delete[] checkZeros;

		//	return count;
		//}
    }
}
