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

        #define blockSize 128

        __global__ void kernMapToBoolean(int N, int* arr, int* boolArr)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) return;

            boolArr[index] = arr[index];
            if (boolArr[index] != 0)
            {
                boolArr[index] = 1;
            }
        }

        __global__ void kernUpSweep(int N, int d, int* arr)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) return;

            int powDPlus = 1 << (d+1);
            int powD = 1 << d;

            if (index % powDPlus == 0)
            {
                arr[index + powDPlus - 1] += arr[index + powD - 1];
            }
            if (index == N - 1)
            {
                arr[index] = 0;
            }
        }

        __global__ void kernDownSweep(int N, int d, int* arr)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) return;

            int powDPlus = 1 << (d + 1);
            int powD = 1 << d;

            if (index % powDPlus == 0)
            {
                int temp = arr[index + powD - 1];
                arr[index + powD - 1] = arr[index + powDPlus - 1];
                arr[index + powDPlus - 1] += temp;
            }
        }

        __global__ void kernInclusiveToExclusive(int N, int* arr)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) return;

            arr[index] -= arr[0];
        }

        __global__ void kernScatter(int N, int* idata, int* boolArr, int* scanArr, int* odata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) return;

            if (boolArr[index] == 1)
            {
                int idx = scanArr[index];
                odata[idx] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int pow2Length = 1 << ilog2ceil(n);
            int* idataPow2 = new int[pow2Length];
            memcpy(idataPow2, idata, n * sizeof(int));
            for (int i = n; i < pow2Length; i++)
            {
                idataPow2[i] = 0;
            }

            int* dev_arr;

            cudaMalloc((void**)&dev_arr, pow2Length * sizeof(int));
            cudaMemcpy(dev_arr, idataPow2, sizeof(int) * n, cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();
            
            for (int i = 0; i < ilog2ceil(n); i++)
            {
                kernUpSweep << <pow2Length, blockSize >> > (pow2Length, i, dev_arr);
            }
            
            for (int j = ilog2ceil(n)-1; j >= 0; j--)
            {
                kernDownSweep << <pow2Length, blockSize >> > (pow2Length, j, dev_arr);
            }

            kernInclusiveToExclusive << <pow2Length, blockSize >> > (pow2Length, dev_arr);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_arr, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_arr);
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
            int* dev_boolArr;
            int* dev_scanArr;
            int* dev_idata;
            int* dev_odata;

            int* host_boolArr = new int[n];
            int* host_scanArr = new int[n];

            cudaMalloc((void**)&dev_boolArr, n * sizeof(int));
            cudaMalloc((void**)&dev_scanArr, n * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            
            bool standalone = true;
            try { timer().startCpuTimer(); }
            catch (std::exception) { standalone = false; }
            
            kernMapToBoolean << <n, blockSize >> > (n, dev_idata, dev_boolArr);
            cudaMemcpy(host_boolArr, dev_boolArr, sizeof(int) * n, cudaMemcpyDeviceToHost);

            scan(n, host_scanArr, host_boolArr);
            cudaMemcpy(dev_scanArr, host_scanArr, sizeof(int) * n, cudaMemcpyHostToDevice);

            kernScatter << <n, blockSize >> > (n, dev_idata, dev_boolArr, dev_scanArr, dev_odata);

            if (standalone) { timer().endCpuTimer(); }

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            int num = 0;
            for (int i = n-1; i >= 0; i--)
            {
                if (host_boolArr[i] != 0)
                {
                    num = host_scanArr[i] + 1;
                    break;
                }
            }

            cudaFree(dev_boolArr);
            cudaFree(dev_scanArr);
            cudaFree(dev_idata);
            cudaFree(dev_odata);

            free(host_boolArr);
            free(host_scanArr);

            return num;
        }
    }
}
