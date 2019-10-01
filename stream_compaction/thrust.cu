#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
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
			// Declare on the host and cast
			thrust::host_vector<int> h_in(idata, idata + n);
			thrust::host_vector<int> h_out(idata, idata + n);

			thrust::device_vector<int> dv_in = h_in;
			thrust::device_vector<int> dv_out = h_out;

			// START
            timer().startGpuTimer();
            // Thrust it
            thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
			// END
            timer().endGpuTimer();

			// Copy to odata
			thrust::copy(dv_out.begin(), dv_out.end(), odata);
        }

		void radix(int n, int *odata, const int *idata) {
			// Declare on the host and cast
			thrust::host_vector<int> h_in(idata, idata + n);
			thrust::host_vector<int> h_out(idata, idata + n);

			thrust::device_vector<int> dv_in = h_in;
			thrust::device_vector<int> dv_out = h_out;

			// START
			timer().startGpuTimer();
			// Thrust it
			thrust::sort(dv_in.begin(), dv_in.end());
			// END
			timer().endGpuTimer();

			// Copy to odata
			thrust::copy(dv_in.begin(), dv_in.end(), odata);
		}
    }
}
