#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

		void my_scan(int n, int *odata, const int *idata) {
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				odata[i] = odata[i - 1] + idata[i - 1];
			}
		}

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
		//compute an exclusive prefix sum
        void scan(int n, int *odata, const int *idata) {
			//odata is b, idata is a 
	        timer().startCpuTimer();
            // TODO
			my_scan(n, odata, idata);
	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
		//remove 0
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            // TODO
			int index = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[index] = idata[i];
					index++;
				}
			}
	        timer().endCpuTimer();
            return index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        // TODO
			//map
			int* map_data = new int[n];
			for (int i = 0; i < n; i++) {
				if (idata[i] == 0) {
					map_data[i] = 0;
				} else {
					map_data[i] = 1;
				}
			}
			//scan
			int* scan_out = new int[n];
			//the last index of each number is the non-zero index
			my_scan(n, scan_out, map_data);
			//scatter
			int count = 0;
			for (int i = 0; i < n; i++) {
				if (map_data[i] == 1) {
					odata[scan_out[i]] = idata[i];
					count++;
				}
			}
	        timer().endCpuTimer();
            return count;
        }
    }
}
