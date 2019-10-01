#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "naive.h"
#include "radix.h"

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernGetEArray(int n, int bit, int *eArray, const int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n) {
				eArray[index] = !((idata[index] & (1 << bit)) >> bit);
			}
		}

		__global__ void kernGetTArray(int n, int totalFalses, int *tArray, const int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n) {
				tArray[index] = index - idata[index]  + totalFalses;
			}
		}

		__global__ void kernDScatter(int n, int *odata, const int *tArray,
			const int *fArray, const int *eArray, const int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n) {

				int dVal = (!eArray[index] ? tArray[index] : fArray[index]);
				odata[dVal] = idata[index];
			}
		}

		__global__ void kernArraySplice(int n, int mid, 
			int *oLeft, int *oRight, const int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index < mid) {
				oLeft[index] = idata[index];

			} else if (index < n) {
				oRight[index - mid] = idata[index];
			}
		}

		__global__ void kernArrayMerge(int n, int mid,
			int *odata, int *iLeft, const int *iRight) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index < mid) {
				odata[index] = iLeft[index];

			}
			else if (index < n) {
				odata[index] = iRight[index - mid];
			}
		}



		int *splitRecurse(int n, int bit, const int *idata) {
			if (bit >= 0) {
				// Arrays to Use:
				int* dev_idata;		 // Device copy of idata
				int* dev_eArray;	 // 0-1 vers of idata, memcpy'ed into iEData
				int* dev_fArray;	 // Post-scan output, device version
				int* dev_tArray;	 // Array based on total falses
				int* dev_odata;		 // Output result - scattered "d" array

				int* iEData = new int[n]; // Host version of eArray
				int* oFData = new int[n]; // scan result

				// Allocate memory
				cudaMalloc((void**)&dev_idata, n * sizeof(int));
				checkCUDAErrorFn("cudaMalloc dev_idata failed!", "efficient.cu", 81);

				cudaMalloc((void**)&dev_eArray, n * sizeof(int));
				checkCUDAErrorFn("cudaMalloc dev_eArray failed!", "efficient.cu", 84);

				cudaMalloc((void**)&dev_fArray, n * sizeof(int));
				checkCUDAErrorFn("cudaMalloc dev_fArray failed!", "efficient.cu", 87);

				cudaMalloc((void**)&dev_tArray, n * sizeof(int));
				checkCUDAErrorFn("cudaMalloc dev_tArray failed!", "efficient.cu", 90);

				cudaMalloc((void**)&dev_odata, n * sizeof(int));
				checkCUDAErrorFn("cudaMalloc dev_odata failed!", "efficient.cu", 93);

				// Copying the input array to the device
				cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
				checkCUDAErrorFn("cudaMemcpy to input-copy failed!", "efficient.cu", 97);


				// KERN TIME
				// Compute temporary array, e, of 0's and 1's
				kernGetEArray << < n, blockSize >> > (n, bit, dev_eArray, dev_idata);

				// Move this data back to host and pass into scan(), then copy data back
				cudaMemcpy(iEData, dev_eArray, n * sizeof(int), cudaMemcpyDeviceToHost);
				checkCUDAErrorFn("cudaMemcpy to host e failed!", "efficient.cu", 107);

				// This is to use my efficiennt implementation of scan, which is exclusive
				StreamCompaction::Efficient::scan(n, oFData, iEData);
				// This is to use an inclusive scan, which is what Radix expects
				//StreamCompaction::Naive::scanInclusive(n, oFData, iEData);

				cudaMemcpy(dev_fArray, oFData, n * sizeof(int), cudaMemcpyHostToDevice);
				checkCUDAErrorFn("cudaMemcpy to f array failed!", "efficient.cu", 115);

				// Compute t array
				int totalFalses = iEData[n - 1] + oFData[n - 1];
				kernGetTArray << < n, blockSize >> > (n, totalFalses, dev_tArray, dev_fArray);

				// Pass all data into scatter
				kernDScatter << < n, blockSize >> > (n, dev_odata, dev_tArray,
					dev_fArray, dev_eArray, dev_idata);

				// Different possibilities:
				int* result = new int[n];
				if (bit == 0) {
					//int* odata = new int[n];
					// Copying the result back to the output array
					cudaMemcpy(result, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
					checkCUDAErrorFn("cudaMemcpy to output failed!", "efficient.cu", 127);

					// SELECT RESULT
					//result = odata;
				}
				else if (totalFalses != 0 && (n - totalFalses) != 0) {
					// SPLIT starting at index totalFalse
					// -First create arrays of new lengths
					int* dev_lArray;
					int* dev_rArray;

					// -Allocate
					cudaMalloc((void**)&dev_lArray, totalFalses * sizeof(int));
					checkCUDAErrorFn("cudaMalloc dev_lArray failed!", "efficient.cu", 140);

					cudaMalloc((void**)&dev_rArray, (n - totalFalses) * sizeof(int));
					checkCUDAErrorFn("cudaMalloc dev_rArray failed!", "efficient.cu", 143);


					kernArraySplice << < n, blockSize >> > (n, totalFalses,
						dev_lArray, dev_rArray, dev_odata);


					// -Copying the two arrays back to host
					int* idataFalse = new int[totalFalses];
					int* idataTrue = new int[n - totalFalses];

					cudaMemcpy(idataFalse, dev_lArray, totalFalses * sizeof(int), cudaMemcpyDeviceToHost);
					checkCUDAErrorFn("cudaMemcpy to odataFalse failed!", "efficient.cu", 155);

					cudaMemcpy(idataTrue, dev_rArray, (n - totalFalses) * sizeof(int), cudaMemcpyDeviceToHost);
					checkCUDAErrorFn("cudaMemcpy to odataFalse failed!", "efficient.cu", 158);

					// RECURSE
					int* lToMerge = splitRecurse(totalFalses, bit - 1, idataFalse);
					int* rToMerge = splitRecurse(n - totalFalses, bit - 1, idataTrue);

					// Pass back to device :(
					cudaMemcpy(dev_lArray, lToMerge, totalFalses * sizeof(int), cudaMemcpyHostToDevice);
					checkCUDAErrorFn("cudaMemcpy to left-recurse-copy failed!", "efficient.cu", 166);

					cudaMemcpy(dev_rArray, rToMerge, (n - totalFalses) * sizeof(int), cudaMemcpyHostToDevice);
					checkCUDAErrorFn("cudaMemcpy to right-recurse-copy failed!", "efficient.cu", 169);

					// MERGE
					kernArrayMerge << < n, blockSize >> > (n, totalFalses,
						dev_odata, dev_lArray, dev_rArray);

					// Copying the result back to the output array
					cudaMemcpy(result, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
					checkCUDAErrorFn("cudaMemcpy to output failed!", "efficient.cu", 178);

					cudaFree(dev_lArray);
					cudaFree(dev_rArray);
				}
				else if (totalFalses == 0 || (n - totalFalses) == 0) {
					int* odata = new int[n];
					// Copying the result back to the output array
					cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
					checkCUDAErrorFn("cudaMemcpy to output failed!", "efficient.cu", 190);

					// SELECT RESULT
					//result = splitRecurse(n, bit - 1, odata);
					cudaMemcpy(result, splitRecurse(n, bit - 1, odata), n * sizeof(int), cudaMemcpyHostToHost);

					delete[] odata;
				}
				else {
					result = {};
				}

				cudaFree(dev_idata);
				cudaFree(dev_eArray);
				cudaFree(dev_fArray);
				cudaFree(dev_tArray);
				cudaFree(dev_odata);

				delete[] iEData;
				delete[] oFData;

				return result;
			} 
			else {
				//int* odata = new int[n];
				//cudaMemcpy(odata, idata, n * sizeof(int), cudaMemcpyHostToHost);
				return {};
			}
		}

        /**
         * Performs radix sort on idata, storing the result into odata.
         */
        void sort(int n, int *odata, const int *idata) {
			// START
			timer().startGpuTimer();
			StreamCompaction::Efficient::changeTimeBool(true);

			// Recursive Radix Sort
			cudaMemcpy(odata, splitRecurse(n, 7, idata), n * sizeof(int), cudaMemcpyHostToHost);
			//odata = splitRecurse(n, 3, idata);
			
			// END
			timer().endGpuTimer();
			StreamCompaction::Efficient::changeTimeBool(false);

/*
			// Arrays to Use:
			int* dev_idata;		 // Device copy of idata
			int* dev_eArray;	 // 0-1 vers of idata, memcpy'ed into iEData
			int* dev_fArray;	 // Post-scan output, device version
			int* dev_tArray;	 // Array based on total falses
			int* dev_odata;		 // Output result - scattered "d" array

			int* iEData = new int[n]; // Host version of eArray
			int* oFData = new int[n]; // scan result

			// Allocate memory
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_idata failed!", "efficient.cu", 53);

			cudaMalloc((void**)&dev_eArray, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_eArray failed!", "efficient.cu", 56);

			cudaMalloc((void**)&dev_fArray, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_fArray failed!", "efficient.cu", 53);

			cudaMalloc((void**)&dev_tArray, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_tArray failed!", "efficient.cu", 53);

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_odata failed!", "efficient.cu", 56);

			// Copying the input array to the device
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy to input-copy failed!", "efficient.cu", 60);


			// START
			timer().startGpuTimer();
			StreamCompaction::Efficient::changeTimeBool(true);

			// Compute temporary array, e, of 0's and 1's
			float bit = 3; // make loop [0,4)
			kernGetEArray << < n, blockSize >> > (n, bit, dev_eArray, dev_idata);

			// Move this data back to host and pass into scan(), then copy data back
			cudaMemcpy(iEData, dev_eArray, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy to host e failed!", "efficient.cu", 81);

			StreamCompaction::Efficient::scan(n, oFData, iEData);

			cudaMemcpy(dev_fArray, oFData, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy to f array failed!", "efficient.cu", 60);

			// Compute t array
			int totalFalses = iEData[n - 1] + oFData[n - 1];
			kernGetTArray << < n, blockSize >> > (n, totalFalses, dev_tArray, dev_fArray);

			// Pass all data into scatter
			kernDScatter << < n, blockSize >> > (n, dev_odata, dev_tArray, 
				dev_fArray, dev_eArray, dev_idata);

			//dev_idata = dev_odata;

			// Split starting at index totalFalses? and Repeat
			// Then merge

			// END
			timer().endGpuTimer();
			StreamCompaction::Efficient::changeTimeBool(false);


			// Copying the result back to the output array
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy to output failed!", "efficient.cu", 81);

			cudaFree(dev_idata);
			cudaFree(dev_eArray);
			cudaFree(dev_fArray);
			cudaFree(dev_tArray);
			cudaFree(dev_odata);

			delete[] iEData;
			delete[] oFData; 

*/

        }
    }
}
