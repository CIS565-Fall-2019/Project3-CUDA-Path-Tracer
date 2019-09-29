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

        void scan(int n, int *odata, const int *idata) {
            
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

			thrust::host_vector<int>host_idata(idata, idata+n);
			thrust::host_vector<int>host_odata(odata, odata+n);
			checkCUDAErrorFn("thrust::host_vector host_odata or host_idata failed!");
			printf("Created Thrust pointers \n");

			thrust::device_vector<int> device_idata = host_idata;
			thrust::device_vector<int> device_odata = host_odata;
			checkCUDAErrorFn("thrust::device_vector device_idata or device_odata failed!");

			timer().startGpuTimer();
			thrust::exclusive_scan(device_idata.begin(), device_idata.end(), device_odata.begin());
			timer().endGpuTimer();

			// Copy back to cpu
			thrust::copy(device_odata.begin(), device_odata.end(), odata);
        }
    }
}
