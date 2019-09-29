#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
		//https://thrust.github.io/doc/group__prefixsums_ga7be5451c96d8f649c8c43208fcebb8c3.html
        void scan(int n, int *odata, const int *idata) {
			thrust::host_vector<int> temp(idata, idata + n);
			thrust::device_vector<int> dev_in(temp);
			thrust::device_vector<int> dev_out(n);
			cudaDeviceSynchronize();
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            //thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
			thrust::exclusive_scan(dev_in.begin(), dev_in.end(), dev_out.begin());
            timer().endGpuTimer();

			cudaMemcpy(odata, thrust::raw_pointer_cast(&dev_out[0]), sizeof(int) * n, cudaMemcpyDeviceToHost);//get the result
		//	checkCUDAError("get odata failed!");
        }
    }
}
