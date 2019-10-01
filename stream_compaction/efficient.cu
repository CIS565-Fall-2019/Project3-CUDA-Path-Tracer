#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
		bool timeInProg = false;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		// **SCAN KERNS** Using textbook and slide formulas
		__global__ void kernScanUpSweep(int n, int d, int *idata) {     // BEFORE Warp Partitioning:
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;		// int index = (blockIdx.x * blockDim.x) + threadIdx.x;
																		// if (index < n) {
			int exp = (int) powf(2.f, float(d) + 1.f);					//	   int exp = (int)powf(2.f, float(d) + 1.f);
			int offset = index * exp + exp - 1;							//     if (index % exp == 0) {
																		//         idata[index + exp - 1] += idata[index + (exp / 2) - 1];
			if (offset < n) {											//	   }
				idata[offset] += idata[offset - exp / 2];				// }
			}
		}

		__global__ void kernSetRootZero(int n, int *idata) { // Had to be seperate from Down
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index == n - 1) {
				idata[index] = 0;
			}
		}

		__global__ void kernScanDownSweep(int n, int d, int *idata) {	// BEFORE Warp Partitioning:
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;		// int index = (blockIdx.x * blockDim.x) + threadIdx.x;
																		// if (index < n) {
			int exp = (int)powf(2.f, float(d) + 1.f);					//	   int exp = (int)powf(2.f, float(d) + 1.f);
			int offset = index * exp + exp - 1;
																		//	   if (index % exp == 0) {
			if (offset < n) {											//		   int t = idata[index + (exp / 2) - 1];
				int t = idata[offset - exp / 2]; 						//		   idata[index + (exp / 2) - 1] = idata[index + exp - 1];
				idata[offset - exp / 2] = idata[offset];				//		   idata[index + exp - 1] += t;
				idata[offset] += t;										//	   }
			}															// }
		}


		// **COMPACT KERNS**
		__global__ void kernMapTo01(int n, int *data01, const int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n) {
				data01[index] = (idata[index] == 0 ? 0 : 1);
			}
		}

		__global__ void kernScatter(int n, int *odata, const int *scanResult,
			const int *data01, const int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n) {

				if (data01[index] == 1) {
					odata[scanResult[index]] = idata[index];
				}
			}
		}

		// For radix sort
		void changeTimeBool(bool time) {
			timeInProg = time;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int log2 = ilog2ceil(n);
			int newLength = (int) pow(2, log2);

			int* idataExtend = new int[newLength];

			// Copying info from idata
			for (int i = 0; i < n; i++) {
				idataExtend[i] = idata[i];
			}
			for (int i = n; i < newLength; i++) {
				idataExtend[i] = 0;
			}

			int* dev_idata;
			cudaMalloc((void**)&dev_idata, newLength * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_idata failed!", "efficient.cu", 93);

			// Copying the input array to the device
			cudaMemcpy(dev_idata, idataExtend, newLength * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy to input-copy failed!", "efficient.cu", 97);

			// START
			if (!timeInProg) { timer().startGpuTimer(); }

			// Up Sweep
			for (int d = 0; d <= log2 - 1; d++) {
				kernScanUpSweep << < newLength, blockSize >> > (newLength, d, dev_idata);
				checkCUDAErrorFn("kernScanUpSweep failed!", "efficient.cu", 105);
			}
			// Setting root to zero
			kernSetRootZero << < newLength, blockSize >> > (newLength, dev_idata);
			// Down Sweep
			for (int d = log2 - 1; d >= 0; d--) {
				kernScanDownSweep << < newLength, blockSize >> > (newLength, d, dev_idata);
				checkCUDAErrorFn("kernScanDownSweep failed!", "efficient.cu", 112);
			}

			// END
			if (!timeInProg) { timer().endGpuTimer(); }

			// Copying the result back to the output array
			cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy to output failed!", "efficient.cu", 120);

			delete[] idataExtend;

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
			// Arrays to Use:
			int* dev_idata;		 // Device copy of idata
			int* dev_binaryMap;  // 0-1 vers of idata, memcpy'ed into iScanData
			int* dev_scanResult; // Device copy of post-scan binary map (oScanData)
			int* dev_odata;		 // Output result

			int* iScanData = new int[n]; // 0-1 input to scan()
			int* oScanData = new int[n]; // scan result

			// Allocate memory
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_idata failed!", "efficient.cu", 148);

			cudaMalloc((void**)&dev_binaryMap, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_binaryMap failed!", "efficient.cu", 151);

			cudaMalloc((void**)&dev_scanResult, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_scanResult failed!", "efficient.cu", 154);

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_odata failed!", "efficient.cu", 157);

			// Copying the input array to the device
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy to input-copy failed!", "efficient.cu", 161);


			// START
			timer().startGpuTimer();
			timeInProg = true;

			// Compute temporary array of 0's and 1's
			kernMapTo01 << < n, blockSize >> > (n, dev_binaryMap, dev_idata);

			// Move this data back to host and pass into scan(), then copy data back
			cudaMemcpy(iScanData, dev_binaryMap, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy to 01-scan failed!", "efficient.cu", 173);

			scan(n, oScanData, iScanData);

			cudaMemcpy(dev_scanResult, oScanData, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy to scanResult failed!", "efficient.cu", 178);

			// Pass data into scatter 
			kernScatter << < n, blockSize >> > (n, dev_odata, dev_scanResult,
				dev_binaryMap, dev_idata);

			// END
			timer().endGpuTimer();
			timeInProg = false;


			// Finding new length
			int newLength = (iScanData[n - 1] == 0 ? oScanData[n - 1] : oScanData[n - 1] + 1);

			// Copying the result back to the output array
			cudaMemcpy(odata, dev_odata, newLength * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy to output failed!", "efficient.cu", 194);

			cudaFree(dev_idata);
			cudaFree(dev_binaryMap);
			cudaFree(dev_scanResult);
			cudaFree(dev_odata);

			delete[] iScanData;
			delete[] oScanData;

            return newLength;
        }
    }
}
