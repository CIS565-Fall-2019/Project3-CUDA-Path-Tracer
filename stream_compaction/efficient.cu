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

		// TODO: __global__
		__global__ void kernUpSweep(int n, int d, int *data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			int offset = (int)(powf(2.0, 1.0*(d + 1))) - 1;//2^(d+1) - 1
			int offset2 = (int)(powf(2.0, 1.0*d)) - 1;//2^d-1
			if (index % (offset + 1) == 0) {
				data[index + offset] += data[index + offset2];
			}
		}
		//only get the work index and do 
		__global__ void kernUpSweep2(int n, int time, int* data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			int i = (index + 1) * time - 1;
			int j = i - time / 2;
			data[i] += data[j];
		}

		__global__ void kernDownSweep(int n, int d, int *data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			//do the naive scan
			int offset = (int)(powf(2.0, 1.0 * (d + 1))) - 1;//2^(d+1) - 1
			int offset2 = (int)(powf(2.0, 1.0 * d)) - 1;//2^d-1
			if (index % (offset + 1) == 0) {
				int t = data[index + offset2];
				data[index + offset2] = data[index + offset];
				data[index + offset] += t;
			}
		}

		__global__ void kernDownSweep2(int n, int time, int* data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			int i = (index + 1) * time - 1;
			int j = i - time / 2;
			int t = data[i - time / 2];
			data[j] = data[i];
			data[i] += t;
		}


		//judge is pow of 2? 
		//https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
		bool IsPowerOfTwo(int number) {
			if (number == 0)
				return false;
			for (int power = 1; power > 0; power = power << 1) {
				// This for loop used shifting for powers of 2, meaning
				// that the value will become 0 after the last shift
				// (from binary 1000...0000 to 0000...0000) then, the 'for'
				// loop will break out.
				if (power == number)
					return true;
				if (power > number)
					return false;
			}
			return false;
		}

		__global__ void reviseElemnt(int index, int * data, int value) {
			data[index] = value;
		}

		__global__ void Init(int n, int *data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			data[index] = 0;
		}


		void my_scan(dim3 fullBlocksPerGrid, int pow2n, int d, int *dev_data) {
			//up
			for (int i = 0; i <= d; i++) {
				kernUpSweep << <fullBlocksPerGrid, blockSize >> > (pow2n, i, dev_data);
			}
			reviseElemnt << <dim3(1), dim3(1) >> > (pow2n - 1, dev_data, 0);
			//down
			for (int i = d; i >= 0; i--) {
				kernDownSweep << <fullBlocksPerGrid, blockSize >> > (pow2n, i, dev_data);
			}
		}

		bool time = false;
		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {
			//TODO
			int *dev_data;
			bool is_pow2 = IsPowerOfTwo(n);
			int d = ilog2ceil(n) - 1;//log2 n
			int pow2n = n;
			if (!is_pow2) {
				pow2n = static_cast<int>(pow(2.0, 1.0 * (d + 1)));
			}
			dim3 fullBlocksPerGrid((pow2n + blockSize - 1) / blockSize);

			//malloc memory
			cudaMalloc((void**)&dev_data, pow2n * sizeof(int));
			checkCUDAErrorS("cudaMalloc dev_idata failed!");
			Init << <fullBlocksPerGrid, blockSize >> > (pow2n, dev_data);
			//mempy
			cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);//host to device
			checkCUDAErrorS("cudaMemcpy dev_idata failed!");
			//start time 
			if (time) {
				timer().startGpuTimer();
			}
			//my_scan(fullBlocksPerGrid, pow2n, d, dev_data);
			//up
			//before change each block has 128,1,1 threads (128)
			//each grid has (pow2n + blockSize - 1) / blockSize block (2)
			for (int i = 0; i <= d; i++) {
				int new_block = static_cast<int>(powf(2.0f, 1.0 * (ilog2ceil(n) - i - 1)));
				dim3 new_blockpergrid((new_block + blockSize - 1) / blockSize);
				//kernUpSweep << <fullBlocksPerGrid, blockSize >> > (pow2n, i, dev_data);
				int off = powf(2.0, i + 1);
				kernUpSweep2 << <new_blockpergrid, blockSize >> > (new_block, off, dev_data);
				checkCUDAErrorS("up sweep failed!");
			}
			reviseElemnt << <dim3(1), dim3(1) >> > (pow2n - 1, dev_data, 0);
			//down
			for (int i = d; i >= 0; i--) {
				int new_block = static_cast<int>(powf(2.0f, 1.0 * (ilog2ceil(n) - i - 1)));
				dim3 new_blockpergrid((new_block + blockSize - 1) / blockSize);
				int off = powf(2.0, i + 1);
				//kernDownSweep << <fullBlocksPerGrid, blockSize >> > (pow2n, i, dev_data);
				kernDownSweep2 << <new_blockpergrid, blockSize >> > (new_block, off, dev_data);
				checkCUDAErrorS("down sweep failed!");
			}
			if (time) {
				timer().endGpuTimer();
			}
			//end gpu time
			cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);//get the result
			checkCUDAErrorS("get odata failed!");

			//free
			cudaFree(dev_data);
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
			int *map_data;
			int *scan_out;
			int *dev_idata;
			int *dev_odata;

			bool is_pow2 = IsPowerOfTwo(n);
			int d = ilog2ceil(n) - 1;//log2 n
			int pow2n = n;
			if (!is_pow2) {
				pow2n = static_cast<int>(pow(2.0, 1.0 * (d + 1)));
			}
			dim3 fullBlocksPerGrid((pow2n + blockSize - 1) / blockSize);

			//malloc memory
			cudaMalloc((void**)&map_data, pow2n * sizeof(int));
			checkCUDAErrorS("cudaMalloc map_data failed!");
			Init << <fullBlocksPerGrid, blockSize >> > (pow2n, map_data);
			cudaMalloc((void**)&scan_out, pow2n * sizeof(int));
			checkCUDAErrorS("cudaMalloc scan_out failed!");
			Init << <fullBlocksPerGrid, blockSize >> > (pow2n, scan_out);
			cudaMalloc((void**)&dev_idata, pow2n * sizeof(int));
			checkCUDAErrorS("cudaMalloc dev_idata failed!");
			Init << <fullBlocksPerGrid, blockSize >> > (pow2n, dev_idata);
			cudaMalloc((void**)&dev_odata, pow2n * sizeof(int));
			checkCUDAErrorS("cudaMalloc dev_idata failed!");
			Init << <fullBlocksPerGrid, blockSize >> > (pow2n, dev_odata);

			//mempy
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);//host to device
			checkCUDAErrorS("cudaMemcpy dev_idata failed!");

			//timer().startGpuTimer();
			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (pow2n, map_data, dev_idata);
			//scan
			time = false;
			scan(pow2n, scan_out, map_data);
			//scatter here is n, not pow2n
			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, map_data, scan_out);
			//timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);//get the result
			checkCUDAErrorS("get odata failed!");

			int count = -1;
			int count_zero = -1;
			for (int i = 0; i < n; i++) {
				if (odata[i] == 0) {
					count = i;
					count_zero++;
					break;
				}
			}

			if (count_zero == -1) {
				count = n;
			}

			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(map_data);
			cudaFree(scan_out);
			return count;
		}
	}
}
